"""Post-hoc evaluation of a saved checkpoint, in deterministic AND stochastic mode.

Runs the SAME held-out initial conditions twice: once with the deterministic mean
action, once sampling at the checkpoint's own learned exploration noise. This measures
the execution gap on a contact-rich task -- the hypothesis being that the deterministic
controller can perform a sustained contact sequence that the behaviour policy, perturbed
at every one of 23 chunks, usually cannot.

Kept OUT of the training loop deliberately. The SAC arm of the first matched comparison
ran before this existed, so adding a stochastic pass to the RLPD training loop would
have made the two arms differ in evaluation protocol. Running it here, afterwards and
identically on both final checkpoints, keeps the comparison matched.

Usage:
    OUTF=<json> CKPT=<path> RUN_DIR=<dir> [N_EVAL=25] [MODES=both]
    python eval_checkpoint.py
"""
import json
import math
import os
import random
import sys
import time
import traceback

import numpy as np

OUT = os.environ["OUTF"]
CKPT_PATH = os.environ["CKPT"]
RUN_DIR = os.environ.get("RUN_DIR", "/tmp")
N_EVAL = int(os.environ.get("N_EVAL", "25"))
RESET_SUITE_SEED = int(os.environ.get("RESET_SUITE_SEED", "20260817"))
EP_CHUNKS = int(os.environ.get("EP_CHUNKS", "23"))
SEED = int(os.environ.get("SEED", "0"))
#: "both" (default) runs deterministic+stochastic; "det" runs deterministic only.
#: The comparison metric is the deterministic policy, so a det-only pass halves cost
#: when upgrading many saved checkpoints to the larger evaluation suite.
MODES = os.environ.get("MODES", "both").lower()
#: Control steps over which a new chunk ramps in from the previous chunk's last command.
#: 0 (the default) is the study's execution, unchanged. Non-zero is a DIFFERENT EXECUTION
#: MODE -- see docs/contracts/g1_piston_chunk_boundary_jitter.json -- and its results must
#: be reported as a separate arm, never merged into the frozen comparison.
BLEND_STEPS = int(os.environ.get("BLEND_STEPS", "0"))
#: Demonstration-envelope action filter cutoff (Hz). 0 disables (the study's frozen
#: execution). Non-zero is a DIFFERENT EXECUTION MODE -- see
#: docs/contracts/g1_piston_rl_induced_oscillation.json -- reported as its own arm.
FILTER_HZ = float(os.environ.get("FILTER_HZ", "0"))

os.makedirs(RUN_DIR, exist_ok=True)
res = {"checkpoint": CKPT_PATH, "n_eval": N_EVAL, "modes": {},
       "blend_steps": BLEND_STEPS, "filter_hz": FILTER_HZ}


def emit(s="RUNNING"):
    res["_status"] = s
    with open(OUT, "w") as f:
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
    reward_fn = RW.PistonTaskReward(sc, jn)
    act_filter = (AF.DemoEnvelopeFilter(dim=30, fc_hz=FILTER_HZ)
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
        apply_reset_condition,
        build_reset_suite,
    )
    from rlinf.models.embodiment.modules.gaussian_policy import SquashedNormal

    ACTION_LOW, ACTION_HIGH = -2.2, 2.2
    ACT_MASK = RLSP.build_active_mask().to(DEV)
    FROZEN_V = RLSP.load_frozen_values(STATS).to(DEV).float()

    # Load the trained head + exploration state. An absent CKPT means "evaluate the raw
    # SFT policy", which gives the same two-mode comparison for the baseline.
    if CKPT_PATH and CKPT_PATH != "sft":
        ck = torch.load(CKPT_PATH, map_location="cuda", weights_only=False)
        model.action_model.load_state_dict(ck["action_model"])
        actor_logstd = ck["actor_logstd"].to(DEV)
        res["checkpoint_env_steps"] = int(ck.get("env_steps", -1))
        res["checkpoint_grad_updates"] = int(ck.get("grad_updates", -1))
    else:
        actor_logstd = torch.full((30,), math.log(RLSP.TARGET_ENTROPY_STD), device=DEV)
        res["checkpoint_env_steps"] = 0
    model.eval()
    res["action_std_mean"] = float(torch.exp(actor_logstd).mean())
    res["action_std_active_mean"] = float(torch.exp(actor_logstd)[ACT_MASK].mean())
    emit()

    _, EVAL_CONDITIONS = build_reset_suite(n_eval=50, seed=RESET_SUITE_SEED)
    EVAL_CONDITIONS = EVAL_CONDITIONS[:N_EVAL]

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

    def run_chunk(norm_action, prev_cmd=None):
        phys = nrm.denormalize(norm_action.detach().cpu())
        if act_filter is not None:
            # Filter in physical action space, the units the demonstrations are in;
            # the mapper and retargeter then see demonstration-envelope dynamics.
            import torch as _tf
            phys = _tf.as_tensor(act_filter.filter_chunk(phys.numpy()),
                                 dtype=phys.dtype)
        cmd = retarget.apply(mapper.map(phys).to(env.device), phys.to(env.device))
        if BLEND_STEPS > 0 and prev_cmd is not None:
            blended = CB.blend_chunk(cmd.cpu().numpy(), prev_cmd, BLEND_STEPS)
            cmd = torch.as_tensor(blended, dtype=cmd.dtype, device=cmd.device)
        total_r, done, info = 0.0, False, {}
        for t in range(H):
            a = cmd[t].unsqueeze(0)
            for _hold in range(2):
                _, _, te, tr, _ = env.step(a)
                if bool(te[0]) or bool(tr[0]):
                    done = True
            r, info = reward_fn.step()
            total_r += r
            if done:
                break
        return total_r, done, info, cmd[min(t, H - 1)].cpu().numpy().copy()

    def evaluate(deterministic):
        rows = []
        for cond in EVAL_CONDITIONS:
            env.reset(seed=0)
            apply_reset_condition(env, cond)
            reward_fn.reset()
            _ = act_filter.reset() if act_filter is not None else None
            bar0 = sc["object"].data.body_pos_w[0, 1].cpu().numpy().copy()
            ret, maxlift, stages = 0.0, 0.0, {}
            acts = []
            prev_cmd = None
            for _ in range(EP_CHUNKS):
                img = sc["front_camera"].data.output["rgb"][0].cpu().numpy().copy()
                mean = vlm_encode(img)
                a = sample_action(mean, deterministic)
                acts.append(a.detach().cpu().numpy())
                r, done, info, prev_cmd = run_chunk(a[0], prev_cmd)
                ret += r
                for k, v in info.get("stages", {}).items():
                    stages[k] = stages.get(k, False) or v
                bar = sc["object"].data.body_pos_w[0, 1].cpu().numpy()
                maxlift = max(maxlift, float(bar[2] - bar0[2]))
                if done:
                    break
            bar = sc["object"].data.body_pos_w[0, 1].cpu().numpy()
            av = float(np.var(np.concatenate([x.reshape(-1) for x in acts])))
            rows.append({"condition": cond.index, "hash": cond.hash(),
                         "return": round(float(ret), 3),
                         "disp_m": round(float(np.linalg.norm(bar - bar0)), 4),
                         # Carry/throw test HORIZONTAL transport; disp_m is a 3-D
                         # norm and a vertical fling could clear the threshold on
                         # height alone. Record XY explicitly.
                         "disp_xy_m": round(float(
                             np.linalg.norm((bar - bar0)[:2])), 4),
                         "final_dz_m": round(float(bar[2] - bar0[2]), 4),
                         "max_lift_m": round(maxlift, 4),
                         "action_variance": round(av, 5),
                         "stages": {k: bool(v) for k, v in stages.items()}})
            res["progress"] = {"mode": "det" if deterministic else "stoch",
                               "done": len(rows), "of": len(EVAL_CONDITIONS)}
            emit()

        n = len(rows)

        def rate(k):
            return round(sum(1 for r in rows if r["stages"].get(k)) / n, 3)

        def wilson(k):
            c = sum(1 for r in rows if r["stages"].get(k))
            z, p = 1.96, (c / n if n else 0.0)
            d = 1 + z * z / n
            centre = (p + z * z / (2 * n)) / d
            half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
            return [round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3)]

        return {
            "n_eval_episodes": n,
            "full_success_rate": rate("success"), "reach_rate": rate("reach"),
            "grasp_rate": rate("grasp"), "lift_rate": rate("lift"),
            "plate_rate": rate("plate"), "tube_rate": rate("tube"),
            "ci95": {k: wilson(k) for k in
                     ("success", "reach", "grasp", "lift", "plate", "tube")},
            "mean_return": round(float(np.mean([r["return"] for r in rows])), 3),
            "mean_disp_m": round(float(np.mean([r["disp_m"] for r in rows])), 4),
            "mean_max_lift_m": round(float(np.mean([r["max_lift_m"] for r in rows])), 4),
            "mean_action_variance": round(
                float(np.mean([r["action_variance"] for r in rows])), 5),
            "per_condition": rows,
        }

    res["modes"]["deterministic"] = evaluate(True)
    emit()
    if MODES != "det":
        res["modes"]["stochastic"] = evaluate(False)
        d, s = res["modes"]["deterministic"], res["modes"]["stochastic"]
        res["gap"] = {k: round(d[k] - s[k], 3) for k in
                      ("full_success_rate", "reach_rate", "grasp_rate", "lift_rate",
                       "mean_return", "mean_disp_m")}
    res["wall_clock_s"] = round(time.time() - t_start, 1)
    emit("OK")
    os._exit(0)
except Exception as e:
    res["error"] = f"{type(e).__name__}: {e}"
    res["tb"] = traceback.format_exc()
    emit("FAIL")
    os._exit(1)
