"""Time-boxed throughput benchmark: can we batch online collection across IsaacLab envs?

Measures, for a given num_envs: env decisions/s, episodes/hour, StarVLA batch inference
time, simulator step time, peak VRAM. Uses the existing vectorized IsaacLab path and the
already-batched StarVLA predict_action. No infrastructure rewrite.
"""
import os, sys, json, time, traceback
import numpy as np

OUT = os.environ["OUTF"]
NENV = int(os.environ["NENV"])
N_CHUNKS = int(os.environ.get("N_CHUNKS", "8"))
res = {"num_envs": NENV, "chunks": []}

def emit(s="RUNNING"):
    res["_status"] = s
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2, default=str)

try:
    os.environ.pop("DISPLAY", None)
    sys.path.insert(0, "/home/jren313/research/starvla_rl/starVLA")
    import torch
    from omegaconf import OmegaConf

    CKPT = "/home/jren313/research/starvla_rl/checkpoints/g1-longhorizon-oft-v1/final_model/pytorch_model.pt"
    STATS = "/home/jren313/research/starvla_rl/checkpoints/g1-longhorizon-oft-v1/dataset_statistics.json"
    CFGY = "/home/jren313/research/starvla_rl/checkpoints/g1-longhorizon-oft-v1/config.yaml"
    TASK = "pick up the piston with the right hand, inject it into the tube held by the left hand, then move it over the hole plate."
    RL = "/home/jren313/research/starvla_rl/RLinf/rlinf/envs/isaaclab/tasks/"

    import importlib.util as ilu
    def _load(m, p):
        sp = ilu.spec_from_file_location(m, p); mo = ilu.module_from_spec(sp)
        sys.modules[m] = mo; sp.loader.exec_module(mo); return mo
    Q99 = _load("g1n", RL + "g1_piston_norm.py").Q99ActionNormalizer
    Mapper = _load("g1a", RL + "g1_piston_action.py").G1PistonActionMapper
    HR = _load("g1h", RL + "g1_piston_hand_retarget.py")

    # ---- boot simulator with NENV parallel envs ----
    t0 = time.time()
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=True).app
    # unitree_sdk2py/cyclonedds live in the `isaac` env (same py3.11 ABI); the upstream
    # task package imports DDS at import time, so this must precede `import tasks`.
    sys.path.append("/home/jren313/miniconda3/envs/isaac/lib/python3.11/site-packages")
    sys.path.insert(0, "/home/jren313/unitree_sim_isaaclab")
    import tasks, gymnasium as gym
    from isaaclab_tasks.utils import load_cfg_from_registry
    TID = "Isaac-PickPlace-Piston-G129-Inspire-Joint"
    cfg = load_cfg_from_registry(TID, "env_cfg_entry_point")
    cfg.seed = 0
    cfg.scene.num_envs = NENV
    cfg.scene.left_wrist_camera = None   # front camera only
    cfg.scene.right_wrist_camera = None
    env = gym.make(TID, cfg=cfg, render_mode="rgb_array").unwrapped
    res["sim_boot_s"] = round(time.time() - t0, 1)
    res["actual_num_envs"] = int(env.scene.num_envs)
    jn = list(env.scene["robot"].data.joint_names)
    mapper = Mapper(jn); rt = HR.InspireHandRetargeter(jn)
    res["vram_after_sim_mib"] = int(torch.cuda.memory_allocated() / 1024**2)
    emit()

    # ---- load policy ----
    t0 = time.time()
    mcfg = OmegaConf.load(CFGY)
    from starVLA.model.framework.base_framework import build_framework
    model = build_framework(mcfg)
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(sd, strict=False)
    model = model.to("cuda").eval()
    nrm = Q99.from_dataset_statistics(STATS)
    H = int(model.action_horizon)
    res["model_load_s"] = round(time.time() - t0, 1)
    res["action_horizon"] = H
    res["vram_after_model_mib"] = int(torch.cuda.memory_allocated() / 1024**2)
    emit()

    obs, _ = env.reset(seed=0)

    for c in range(N_CHUNKS):
        # --- batched observation grab ---
        t_obs = time.time()
        rgb = env.scene["front_camera"].data.output["rgb"]  # [NENV,H,W,3]
        rgb_np = rgb.cpu().numpy()
        t_obs = time.time() - t_obs

        # --- ONE batched StarVLA forward across all envs ---
        torch.cuda.synchronize()
        t_inf = time.time()
        with torch.inference_mode():
            out = model.predict_action(
                [{"image": [rgb_np[i]], "lang": TASK} for i in range(NENV)]
            )
        torch.cuda.synchronize()
        t_inf = time.time() - t_inf

        norm = torch.tensor(np.asarray(out["normalized_actions"]), dtype=torch.float32)  # [B,H,30]
        assert norm.shape[0] == NENV, f"policy returned batch {norm.shape[0]} != {NENV}"

        # --- denorm + map + retarget for the whole batch ---
        t_map = time.time()
        phys = torch.stack([nrm.denormalize(norm[i]) for i in range(NENV)])   # [B,H,30]
        cmd = torch.stack([mapper.map(phys[i]) for i in range(NENV)]).to(env.device)  # [B,H,53]
        cmd = rt.apply(cmd, phys.to(env.device))
        t_map = time.time() - t_map

        # --- simulator: H actions x 2 hold, all envs stepped together ---
        t_sim = time.time()
        for t in range(H):
            a = cmd[:, t, :]                      # [B,53]
            for _hold in range(2):
                env.step(a)
        torch.cuda.synchronize()
        t_sim = time.time() - t_sim

        res["chunks"].append({
            "chunk": c, "obs_s": round(t_obs, 4), "infer_s": round(t_inf, 4),
            "map_s": round(t_map, 4), "sim_s": round(t_sim, 4),
            "total_s": round(t_obs + t_inf + t_map + t_sim, 4),
        })
        res["peak_vram_mib"] = int(torch.cuda.max_memory_allocated() / 1024**2)
        emit()

    # ---- aggregate (drop chunk 0: warmup/CUDA graph capture) ----
    ch = res["chunks"][1:] if len(res["chunks"]) > 1 else res["chunks"]
    mean_total = float(np.mean([c["total_s"] for c in ch]))
    # one chunk = H policy actions x 2 hold env steps, per env
    env_steps_per_chunk = H * 2 * NENV
    decisions_per_chunk = H * NENV          # policy actions across all envs
    EP_STEPS = 684                          # measured demo episode length (env steps/env)
    res["summary"] = {
        "mean_chunk_s": round(mean_total, 3),
        "mean_infer_s": round(float(np.mean([c["infer_s"] for c in ch])), 3),
        "mean_sim_s": round(float(np.mean([c["sim_s"] for c in ch])), 3),
        "mean_obs_s": round(float(np.mean([c["obs_s"] for c in ch])), 4),
        "mean_map_s": round(float(np.mean([c["map_s"] for c in ch])), 4),
        "env_steps_per_s": round(env_steps_per_chunk / mean_total, 1),
        "decisions_per_s": round(decisions_per_chunk / mean_total, 2),
        "episodes_per_hour": round(3600.0 / (mean_total * (EP_STEPS / (H * 2))) * NENV, 1),
        "peak_vram_mib": res.get("peak_vram_mib"),
    }
    emit("OK"); os._exit(0)
except Exception as e:
    res["error"] = f"{type(e).__name__}: {e}"
    res["tb"] = traceback.format_exc()
    emit("FAIL"); os._exit(1)
