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
CONDS = [c for c in CONDS for _ in range(REPEATS)]
MODE = os.environ.get("MODE", "deterministic").lower()
TAG = os.environ.get("TAG", "run")
SEED_LABEL = os.environ.get("SEED_LABEL", "?")
FPS = int(os.environ.get("FPS", "20"))
RESET_SUITE_SEED = int(os.environ.get("RESET_SUITE_SEED", "20260817"))
EP_CHUNKS = int(os.environ.get("EP_CHUNKS", "23"))
SEED = int(os.environ.get("SEED", "0"))

os.makedirs(OUTDIR, exist_ok=True)
STATUS = os.path.join(OUTDIR, f"_render_{TAG}_{MODE}.json")
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
    RLSP = _load("g1s", RL + "g1_piston_rl_space.py")

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
    reward_fn = RW.PistonTaskReward(sc, jn)

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

    if CKPT_PATH and CKPT_PATH != "sft":
        ck = torch.load(CKPT_PATH, map_location="cuda", weights_only=False)
        model.action_model.load_state_dict(ck["action_model"])
        actor_logstd = ck["actor_logstd"].to(DEV)
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
        return mean

    def sample_action(mean, deterministic):
        b, c, d = mean.shape
        flat = mean.reshape(b * c, d)
        if deterministic:
            scale = (ACTION_HIGH - ACTION_LOW) / 2.0
            shift = (ACTION_HIGH + ACTION_LOW) / 2.0
            a = torch.tanh(flat) * scale + shift
        else:
            std = torch.exp(actor_logstd).view(1, -1).expand_as(flat)
            a = SquashedNormal(flat, std, low=ACTION_LOW, high=ACTION_HIGH).rsample()
        a = torch.where(ACT_MASK, a, FROZEN_V.expand_as(a))
        return a.reshape(b, c, d)

    def run_chunk(norm_action, frames, grab):
        """Identical to eval_checkpoint.run_chunk, plus frame capture inside the loop."""
        phys = nrm.denormalize(norm_action.detach().cpu())
        cmd = retarget.apply(mapper.map(phys).to(env.device), phys.to(env.device))
        total_r, done, info = 0.0, False, {}
        for t in range(H):
            a = cmd[t].unsqueeze(0)
            for _hold in range(2):
                _, _, te, tr, _ = env.step(a)
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
            if done:
                break
        return total_r, done, info

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
        bar0 = sc["object"].data.body_pos_w[0, 1].cpu().numpy().copy()
        frames, per_chunk = [], []
        ret, maxlift, stages = 0.0, 0.0, {}
        deterministic = (MODE == "deterministic")

        for c in range(EP_CHUNKS):
            img = sc["front_camera"].data.output["rgb"][0].cpu().numpy().copy()
            mean = vlm_encode(img)
            a = sample_action(mean, deterministic)
            r, done, info = run_chunk(a[0], frames, grab)
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
            "rng_seed": SEED,
            "episode_chunks": len(per_chunk),
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
