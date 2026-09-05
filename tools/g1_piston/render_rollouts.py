"""Render REAL closed-loop simulator rollouts from a saved checkpoint, with proof.

This is a strict derivative of ``eval_checkpoint.py``: the model loading, the reset path,
``sample_action`` and ``run_chunk`` are copied verbatim so a rendered rollout is the same
computation the reported metrics came from. The only additions are (a) frame capture,
(b) a per-chunk state trajectory, and (c) an overlay drawn from real simulator state.

The policy actually controls the simulator while recording. Nothing here replays a
logged action array, animates joint values, or reuses demonstration footage.

Every rollout writes a sidecar JSON carrying the checkpoint path, the frozen condition
hash, the raw trajectory, and the metrics recomputed from that trajectory by the frozen
``g1_piston_metrics`` code. ``verify_rollouts.py`` re-derives the labels independently.

Usage:
    OUTDIR=<dir> CKPT=<path|sft> CONDS=0,3,20 [MODE=deterministic|stochastic]
    [TAG=sac_s2] [SEED_LABEL=2] [FPS=20] python render_rollouts.py
"""
import json
import math
import os
import random
import sys
import time
import traceback

import numpy as np

OUTDIR = os.environ["OUTDIR"]
CKPT_PATH = os.environ["CKPT"]
CONDS = [int(x) for x in os.environ["CONDS"].split(",") if x.strip() != ""]
#: Draws per condition. A stochastic rollout samples fresh noise, and on this task WHICH
#: condition succeeds does not replicate across draws while the success RATE does (see
#: docs/contracts/g1_piston_sft_stochastic_success.json), so repeated draws on one
#: condition are the only honest way to show the behaviour rather than one lucky sample.
#: Repeating the condition list is equivalent to an inner loop and needs no restructuring;
#: filenames are disambiguated by the draw index below.
REPEATS = int(os.environ.get("REPEATS", "1"))
#: Disable frame capture, for isolating its effect on contact physics.
NO_CAPTURE = os.environ.get("NO_CAPTURE", "0") == "1"
#: Keep rolling after the simulator signals termination. Display/diagnosis only: the
#: sidecar records where the signal fired (`sim_done_chunk`) so nothing is hidden --
#: it lets the video show what the policy does for the full episode, and what actually
#: happened to the object around the termination.
IGNORE_DONE = os.environ.get("IGNORE_DONE", "0") == "1"
#: Log per-step object pose and termination flags without changing rollout behaviour.
LOG_STEPS = os.environ.get("LOG_STEPS", "0") == "1" or IGNORE_DONE
#: Log the head-camera body pose per step, to separate genuine camera shake (the ego
#: camera rides on d435_link, which moves with the torso) from render artefacts.
LOG_CAMERA = os.environ.get("LOG_CAMERA", "0") == "1"
#: Score the functional pipette task (plunger) instead of transport-only v1.
REWARD_V2 = os.environ.get("REWARD_V2", "0") == "1"
#: Reward v3: review fixes (3-D grasp with opposition, exploit-free success, jerk
#: penalty). Takes precedence over REWARD_V2. See
#: docs/contracts/g1_piston_reward_v3_review_fixes.json.
REWARD_V3 = os.environ.get("REWARD_V3", "0") == "1"
#: Control steps over which a new chunk ramps in from the previous chunk's last command.
#: 0 (the default) reproduces the study's execution exactly. Non-zero is a DIFFERENT
#: EXECUTION MODE and must be reported as its own arm -- see
#: docs/contracts/g1_piston_chunk_boundary_jitter.json.
BLEND_STEPS = int(os.environ.get("BLEND_STEPS", "0"))
#: Demonstration-envelope action filter cutoff (Hz). 0 disables (the study's frozen
#: execution). Non-zero is a DIFFERENT EXECUTION MODE -- see
#: docs/contracts/g1_piston_rl_induced_oscillation.json -- reported as its own arm.
FILTER_HZ = float(os.environ.get("FILTER_HZ", "0"))
#: "all" filters every dim; "hand" only dims 14-25, leaving the arm untouched -- the
#: minimal intervention, since the measured pathology is confined to the hands and
#: whole-action filtering lagged the reach enough to break grasp timing.
FILTER_DIMS = os.environ.get("FILTER_DIMS", "all").lower()
CONDS = [c for c in CONDS for _ in range(REPEATS)]
MODE = os.environ.get("MODE", "deterministic").lower()
TAG = os.environ.get("TAG", "run")
SEED_LABEL = os.environ.get("SEED_LABEL", "?")
FPS = int(os.environ.get("FPS", "20"))
RESET_SUITE_SEED = int(os.environ.get("RESET_SUITE_SEED", "20260817"))
EP_CHUNKS = int(os.environ.get("EP_CHUNKS", "23"))
SEED = int(os.environ.get("SEED", "0"))

os.makedirs(OUTDIR, exist_ok=True)
# The checkpoint MUST be in the status filename. Progression jobs share a tag, mode
# and condition list and differ only by checkpoint; without it they all collide on
# one status file and the queue's is_complete() skips every later checkpoint as
# "already complete" -- which silently gutted the progression sequence once.
_CKPT_ID = ("sft" if CKPT_PATH in ("", "sft")
           else os.path.basename(CKPT_PATH).replace(".pt", ""))
STATUS = os.path.join(OUTDIR, f"_render_{TAG}_{_CKPT_ID}_{MODE}.json")
res = {"tag": TAG, "checkpoint": CKPT_PATH, "mode": MODE, "conditions": CONDS,
       "rollouts": []}


def emit(s="RUNNING"):
    res["_status"] = s
    with open(STATUS, "w") as f:
        json.dump(res, f, indent=2, default=str)


t_start = time.time()
try:
    os.environ.pop("DISPLAY", None)
    sys.path.insert(0, "/home/jren313/research/starvla_rl/starVLA")
    import torch
    from omegaconf import OmegaConf

    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    BASE = "/home/jren313/research/starvla_rl/checkpoints/g1-longhorizon-oft-v1"
    CFGY, STATS = f"{BASE}/config.yaml", f"{BASE}/dataset_statistics.json"
    SFT = f"{BASE}/final_model/pytorch_model.pt"
    TASK = ("pick up the piston with the right hand, inject it into the tube held by "
            "the left hand, then move it over the hole plate.")
    RL = "/home/jren313/research/starvla_rl/RLinf/rlinf/envs/isaaclab/tasks/"

    import importlib.util as ilu

    def _load(m, p):
        sp = ilu.spec_from_file_location(m, p); mo = ilu.module_from_spec(sp)
        sys.modules[m] = mo; sp.loader.exec_module(mo); return mo

    Q99 = _load("g1n", RL + "g1_piston_norm.py").Q99ActionNormalizer
    Mapper = _load("g1a", RL + "g1_piston_action.py").G1PistonActionMapper
    HR = _load("g1h", RL + "g1_piston_hand_retarget.py")
    RW = _load("g1r", RL + "g1_piston_reward.py")
    # REWARD_V2=1 scores the FUNCTIONAL pipette task: the plunger (the object's
    # prismatic PistonJoint) must actually be depressed. v1 scores transport only and
    # stays the default so every prior result remains reproducible. See
    # docs/contracts/g1_piston_plunger_dof.json.
    RW2 = _load("g1r2", RL + "g1_piston_reward_v2.py") if REWARD_V2 else None
    RW3 = _load("g1r3", RL + "g1_piston_reward_v3.py") if REWARD_V3 else None
    RLSP = _load("g1s", RL + "g1_piston_rl_space.py")
    RESP = _load("g1rp", RL + "g1_piston_residual_policy.py")
    CB = _load("g1cb", RL + "g1_piston_chunk_blend.py")
    AF = _load("g1af", RL + "g1_piston_action_filter.py")

    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=True).app
    sys.path.append("/home/jren313/miniconda3/envs/isaac/lib/python3.11/site-packages")
    sys.path.insert(0, "/home/jren313/unitree_sim_isaaclab")
    import gymnasium as gym
    import tasks  # noqa: F401
    from isaaclab_tasks.utils import load_cfg_from_registry

    TID = "Isaac-PickPlace-Piston-G129-Inspire-Joint"
    ecfg = load_cfg_from_registry(TID, "env_cfg_entry_point")
    ecfg.seed = SEED; ecfg.scene.num_envs = 1
    ecfg.scene.left_wrist_camera = None; ecfg.scene.right_wrist_camera = None
    env = gym.make(TID, cfg=ecfg, render_mode="rgb_array").unwrapped
    sc = env.scene
    jn = list(sc["robot"].data.joint_names)
    mapper = Mapper(jn); retarget = HR.InspireHandRetargeter(jn)
    reward_fn = (RW3.PistonTaskRewardV3(sc, jn) if REWARD_V3
                 else RW2.PistonTaskRewardV2(sc, jn) if REWARD_V2
                 else RW.PistonTaskReward(sc, jn))
    act_filter = (AF.DemoEnvelopeFilter(
                      dim=30, fc_hz=FILTER_HZ,
                      apply_dims=(AF.HAND_DIMS if FILTER_DIMS == "hand" else None))
                  if FILTER_HZ > 0 else None)

    mcfg = OmegaConf.load(CFGY)
    from deployment.model_server.tools.image_tools import to_pil_preserve
    from starVLA.model.framework.base_framework import build_framework
    from starVLA.training.trainer_utils.trainer_tools import resize_images

    model = build_framework(mcfg)
    model.load_state_dict(torch.load(SFT, map_location="cpu", weights_only=False),
                          strict=False)
    model = model.to("cuda")
    nrm = Q99.from_dataset_statistics(STATS)
    H = int(model.action_horizon)
    DEV = "cuda"

    sys.path.insert(0, "/home/jren313/research/starvla_rl/RLinf")
    from rlinf.envs.isaaclab.tasks.g1_piston_reset import (
        EXPERIMENT_N_TRAIN,
        apply_reset_condition,
        build_reset_suite,
    )
    from rlinf.models.embodiment.modules.gaussian_policy import SquashedNormal

    ACTION_LOW, ACTION_HIGH = -2.2, 2.2
    ACT_MASK = RLSP.build_active_mask().to(DEV)
    FROZEN_V = RLSP.load_frozen_values(STATS).to(DEV).float()

    residual = None   # set when the checkpoint carries a ResFiT residual
    if CKPT_PATH and CKPT_PATH != "sft":
        ck = torch.load(CKPT_PATH, map_location="cuda", weights_only=False)
        model.action_model.load_state_dict(ck["action_model"])
        actor_logstd = ck["actor_logstd"].to(DEV)
        if "residual" in ck:
            _rc = ck.get("residual_cfg", {})
            residual = RESP.ResidualPolicy(feat_dim=int(_rc.get("feat_dim", 2048)),
                                           hidden=tuple(_rc.get("hidden", (512, 512))),
                                           r_max=float(_rc.get("r_max", RESP.R_MAX_DEFAULT)),
                                           device=DEV)
            residual.load_state_dict(ck["residual"])
            residual.eval()
            res["residual"] = {"applied": True, **{k: (list(v) if isinstance(v, (tuple, list)) else v) for k, v in _rc.items()}}
        res["checkpoint_env_steps"] = int(ck.get("env_steps", -1))
        res["checkpoint_grad_updates"] = int(ck.get("grad_updates", -1))
    else:
        actor_logstd = torch.full((30,), math.log(RLSP.TARGET_ENTROPY_STD), device=DEV)
        res["checkpoint_env_steps"] = 0
        res["checkpoint_grad_updates"] = 0
    model.eval()
    emit()

    # The frozen suite. n_train MUST be the experiment's value: the eval draw depends on
    # it, so a different n_train silently yields different conditions.
    _, EVAL_CONDITIONS = build_reset_suite(n_train=EXPERIMENT_N_TRAIN, n_eval=50,
                                           seed=RESET_SUITE_SEED)
    FROZEN_HASHES = [c.hash() for c in EVAL_CONDITIONS]
    res["suite"] = {"n_train": EXPERIMENT_N_TRAIN, "n_eval": 50,
                    "reset_suite_seed": RESET_SUITE_SEED,
                    "eval_hashes": FROZEN_HASHES}

    def vlm_encode(img):
        with torch.no_grad():
            imgs = [to_pil_preserve([img])]
            size = getattr(model.config.datasets.vla_data, "obs_image_size", None)
            if size:
                imgs = resize_images(imgs, target_size=size)
            toks = model.action_token * model.chunk_len
            instr = TASK + (f" Please predict the next {model.chunk_len} robot actions:"
                            f" <action>{toks}<action>.")
            qi = model.qwen_vl_interface.build_qwenvl_inputs(images=imgs,
                                                             instructions=[instr])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                qo = model.qwen_vl_interface(**qi, output_attentions=False,
                                             output_hidden_states=True, return_dict=True)
                lh = qo.hidden_states[-1]
            aq = model._gather_action_token_embeddings(
                lh, qi.get("input_ids", None), action_token_id=model.action_token_id)
            with torch.autocast("cuda", dtype=torch.float32):
                mean = model.action_model.predict_action(aq.detach().float()).float()
        return mean, aq.detach().float()

    def sample_action(mean, deterministic, aq=None):
        b, c, d = mean.shape
        flat = mean.reshape(b * c, d)
        if residual is not None:
            # ResFiT arm: the base is the frozen head's DETERMINISTIC executed chunk;
            # the residual (and its exploration, if stochastic) composes on top.
            scale = (ACTION_HIGH - ACTION_LOW) / 2.0
            shift = (ACTION_HIGH + ACTION_LOW) / 2.0
            a_base = torch.tanh(flat) * scale + shift
            a_base = torch.where(ACT_MASK, a_base, FROZEN_V.expand_as(a_base)).reshape(b, c, d)
            with torch.no_grad():
                a, _, _ = residual.act(aq, a_base, None if deterministic else actor_logstd,
                                       ACT_MASK, deterministic=deterministic)
            return torch.where(ACT_MASK, a, FROZEN_V.expand_as(a))
        if deterministic:
            scale = (ACTION_HIGH - ACTION_LOW) / 2.0
            shift = (ACTION_HIGH + ACTION_LOW) / 2.0
            a = torch.tanh(flat) * scale + shift
        else:
            std = torch.exp(actor_logstd).view(1, -1).expand_as(flat)
            a = SquashedNormal(flat, std, low=ACTION_LOW, high=ACTION_HIGH).rsample()
        a = torch.where(ACT_MASK, a, FROZEN_V.expand_as(a))
        return a.reshape(b, c, d)

    step_log = []  # per-step object pose + termination flags, IGNORE_DONE only
    cam_log = []   # per-step head-camera pose (LOG_CAMERA)
    _CAM_IDX = list(sc["robot"].data.body_names).index("d435_link")

    def run_chunk(norm_action, frames, grab, prev_cmd=None):
        """Identical to eval_checkpoint.run_chunk, plus frame capture and optional blend.

        Returns the final command issued, so the next chunk can ramp in from it.
        """
        phys = nrm.denormalize(norm_action.detach().cpu())
        if act_filter is not None:
            # Filter in physical action space, the units the demonstrations are in;
            # the mapper and retargeter then see demonstration-envelope dynamics.
            import torch as _tf
            phys = _tf.as_tensor(act_filter.filter_chunk(phys.numpy()),
                                 dtype=phys.dtype)
        cmd = retarget.apply(mapper.map(phys).to(env.device), phys.to(env.device))
        if BLEND_STEPS > 0 and prev_cmd is not None:
            import torch as _t
            blended = CB.blend_chunk(cmd.cpu().numpy(), prev_cmd, BLEND_STEPS)
            cmd = _t.as_tensor(blended, dtype=cmd.dtype, device=cmd.device)
        total_r, done, info = 0.0, False, {}
        for t in range(H):
            a = cmd[t].unsqueeze(0)
            for _hold in range(2):
                _, _, te, tr, _ = env.step(a)
                if LOG_CAMERA:
                    _bp = sc["robot"].data.body_pos_w[0]
                    _bq = sc["robot"].data.body_quat_w[0]
                    cam_log.append(
                        [round(float(x), 6) for x in _bp[_CAM_IDX].cpu().numpy()]
                        + [round(float(x), 6) for x in _bq[_CAM_IDX].cpu().numpy()])
                if LOG_STEPS:
                    step_log.append({
                        "root": [round(float(x), 4) for x in
                                 sc["object"].data.root_pos_w[0].cpu().numpy()],
                        "body1": [round(float(x), 4) for x in
                                  sc["object"].data.body_pos_w[0, 1].cpu().numpy()],
                        "te": bool(te[0]), "tr": bool(tr[0])})
                if bool(te[0]) or bool(tr[0]):
                    done = True
            r, info = reward_fn.step()
            total_r += r
            # Capture every other control step: 50 Hz control -> ~25 fps of real motion,
            # no interpolation and no synthetic frames.
            #
            # NO_CAPTURE=1 disables this entirely, making the rollout byte-identical to
            # eval_checkpoint.run_chunk. Reading front_camera.data.output can drive a
            # sensor/render update mid-chunk, and on a contact-rich task that is a
            # candidate cause of divergence from the stored evaluation. The flag exists
            # to TEST that, by isolating capture as the only difference.
            if t % 2 == 0 and not NO_CAPTURE:
                frames.append(grab())
            if done and not IGNORE_DONE:
                break
        return total_r, done, info, cmd[min(t, H - 1)].cpu().numpy().copy()

    def grab():
        return sc["front_camera"].data.output["rgb"][0].cpu().numpy().copy()[..., :3]

    def overlay(frames, meta):
        """Draw real state values onto the frames. Values come from the trajectory."""
        try:
            from PIL import Image, ImageDraw
        except Exception:
            return frames
        out = []
        for i, f in enumerate(frames):
            im = Image.fromarray(f.astype(np.uint8)).convert("RGB")
            d = ImageDraw.Draw(im)
            k = min(i * len(meta["per_chunk"]) // max(1, len(frames)),
                    len(meta["per_chunk"]) - 1)
            pc = meta["per_chunk"][k]
            lines = [
                f"{meta['method']} s{meta['seed']} | {meta['mode']}",
                f"steps: {meta['env_steps']:,}",
                f"cond {meta['condition']} | {meta['hash'][:8]}",
                f"return {pc['cum_return']:.2f} | stage {pc['stage']}",
                f"disp {pc['disp_m']:.3f}m | lift {pc['max_lift_m']:.3f}m",
            ]
            w = max(d.textlength(t) for t in lines) + 8
            d.rectangle([0, 0, w, 12 * len(lines) + 6], fill=(0, 0, 0))
            for j, t in enumerate(lines):
                d.text((4, 3 + 12 * j), t, fill=(255, 255, 255))
            out.append(np.asarray(im))
        return out

    import imageio.v2 as imageio

    for ci in CONDS:
        cond = EVAL_CONDITIONS[ci]
        env.reset(seed=0)
        apply_reset_condition(env, cond)
        reward_fn.reset()
        _ = act_filter.reset() if act_filter is not None else None
        bar0 = sc["object"].data.body_pos_w[0, 1].cpu().numpy().copy()
        frames, per_chunk = [], []
        sim_done_chunk = None
        step_log.clear()
        cam_log.clear()
        ret, maxlift, stages = 0.0, 0.0, {}
        deterministic = (MODE == "deterministic")
        prev_cmd = None

        for c in range(EP_CHUNKS):
            img = sc["front_camera"].data.output["rgb"][0].cpu().numpy().copy()
            mean, aq = vlm_encode(img)
            a = sample_action(mean, deterministic, aq=aq)
            r, done, info, prev_cmd = run_chunk(a[0], frames, grab, prev_cmd)
            ret += r
            for k, v in info.get("stages", {}).items():
                stages[k] = stages.get(k, False) or v
            bar = sc["object"].data.body_pos_w[0, 1].cpu().numpy()
            maxlift = max(maxlift, float(bar[2] - bar0[2]))
            stage = ("success" if stages.get("success") else
                     "plate" if stages.get("plate") else
                     "tube" if stages.get("tube") else
                     "lift" if stages.get("lift") else
                     "grasp" if stages.get("grasp") else
                     "reach" if stages.get("reach") else "-")
            per_chunk.append({
                "chunk": c, "cum_return": round(float(ret), 3),
                "piston_xyz": [round(float(x), 5) for x in bar],
                "disp_m": round(float(np.linalg.norm(bar - bar0)), 4),
                # Carry/throw test HORIZONTAL transport; disp_m is a 3-D
                # norm and a vertical fling could clear the threshold on
                # height alone. Record XY explicitly.
                "disp_xy_m": round(float(
                    np.linalg.norm((bar - bar0)[:2])), 4),
                "final_dz_m": round(float(bar[2] - bar0[2]), 4),
                "max_lift_m": round(maxlift, 4), "stage": stage,
                "stages": {k: bool(v) for k, v in stages.items()},
            })
            if done:
                if sim_done_chunk is None:
                    sim_done_chunk = c
                if not IGNORE_DONE:
                    break

        bar = sc["object"].data.body_pos_w[0, 1].cpu().numpy()
        row = {"condition": cond.index, "hash": cond.hash(),
               "return": round(float(ret), 3),
               "disp_m": round(float(np.linalg.norm(bar - bar0)), 4),
               # Carry/throw test HORIZONTAL transport; disp_m is a 3-D
               # norm and a vertical fling could clear the threshold on
               # height alone. Record XY explicitly.
               "disp_xy_m": round(float(
                   np.linalg.norm((bar - bar0)[:2])), 4),
               "final_dz_m": round(float(bar[2] - bar0[2]), 4),
               "max_lift_m": round(maxlift, 4),
               "stages": {k: bool(v) for k, v in stages.items()}}

        meta = {"method": TAG, "seed": SEED_LABEL, "mode": MODE,
                "env_steps": res["checkpoint_env_steps"], "condition": cond.index,
                "hash": cond.hash(), "per_chunk": per_chunk}

        name = f"{TAG}_step{res['checkpoint_env_steps']}_cond{cond.index}_{MODE}"
        if REPEATS > 1:
            draw = sum(1 for r in res["rollouts"] if r["cond"] == cond.index)
            name += f"_draw{draw}"
        mp4 = os.path.join(OUTDIR, name + ".mp4")
        try:
            imageio.mimsave(mp4, overlay(frames, meta), fps=FPS,
                            codec="libx264", quality=8)
        except Exception:
            imageio.mimsave(mp4, overlay(frames, meta), fps=FPS)

        sidecar = {
            "video": mp4, "method": TAG, "seed": SEED_LABEL,
            "checkpoint": CKPT_PATH,
            "checkpoint_env_steps": res["checkpoint_env_steps"],
            "checkpoint_grad_updates": res.get("checkpoint_grad_updates"),
            "condition_index": cond.index, "condition_hash": cond.hash(),
            "condition_in_frozen_suite": cond.hash() in FROZEN_HASHES,
            "condition_joint_delta": cond.joint_delta,
            "condition_piston_dxy": list(cond.piston_dxy),
            "reset_suite_seed": RESET_SUITE_SEED,
            "n_train": EXPERIMENT_N_TRAIN,
            "mode": MODE, "fps": FPS, "n_frames": len(frames),
            # A stochastic rollout samples fresh noise. This renderer seeds SEED=0 at
            # start-up, but the trainer's generator had already advanced through a
            # deterministic sweep before its own stochastic pass, so a stochastic render
            # is an INDEPENDENT DRAW from the same policy -- not a replay of the
            # trainer's episode. Its outcome may legitimately differ; the label written
            # here is always recomputed from THIS rollout's trajectory.
            "stochastic_is_independent_draw": (MODE == "stochastic"),
            "blend_steps": BLEND_STEPS,
            "filter_hz": FILTER_HZ,
            "filter_dims": FILTER_DIMS,
            "reward_version": ("v3_review_fixed" if REWARD_V3 else "v2_functional" if REWARD_V2 else "v1_transport"),
            "rng_seed": SEED,
            "episode_chunks": len(per_chunk),
            "sim_done_chunk": sim_done_chunk,
            "step_log": list(step_log) if LOG_STEPS else None,
            "camera_log": list(cam_log) if LOG_CAMERA else None,
            "piston_initial_xyz": [round(float(x), 5) for x in bar0],
            "piston_final_xyz": [round(float(x), 5) for x in bar],
            "row": row, "trajectory": per_chunk,
        }
        with open(os.path.join(OUTDIR, name + ".json"), "w") as f:
            json.dump(sidecar, f, indent=2)
        res["rollouts"].append({"cond": cond.index, "hash": cond.hash(),
                                "video": mp4, "row": row, "frames": len(frames)})
        emit()

    res["wall_clock_s"] = round(time.time() - t_start, 1)
    emit("OK")
    os._exit(0)
except Exception as e:
    res["error"] = f"{type(e).__name__}: {e}"
    res["tb"] = traceback.format_exc()
    emit("FAIL")
    os._exit(1)
