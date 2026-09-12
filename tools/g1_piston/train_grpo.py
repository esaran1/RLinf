#!/usr/bin/env python3
# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Critic-free RL fine-tuning (GRPO + PPO + KL anchor) of a frozen BC head + residual.

See docs/contracts/g1_piston_run9_grpo_preregistration.json for why and for the fixed
constants. The environment, model, normaliser, mapper, retargeter, reward and reset suite
are set up exactly as in train_sac.py; the learning loop is different:

    for each iteration:
        for each of N_GROUPS initial conditions (TRAIN split):
            G stochastic rollouts: a = compose(base, c_mean + sigma*z)
        advantages: per-chunk return-to-go standardised within each group (GRPO)
        PPO epochs over all chunks: clipped ratio on the latent Gaussian log-density
            + KL_BETA * KL(policy || base)   (base = zero residual)
        deterministic evaluation on the fixed EVAL prefix; checkpoint; keep best

Environment: OUTF, RUN_DIR (required); BASE_CKPT; see CONSTANTS below.
"""
import json, os, sys, math, time, random, traceback
import numpy as np

OUT = os.environ["OUTF"]; RUN_DIR = os.environ["RUN_DIR"]; os.makedirs(RUN_DIR, exist_ok=True)
BASE_CKPT = os.environ.get("BASE_CKPT", "/home/jren313/research/starvla_rl/checkpoints/g1_piston_bc_working/bc_ckpt_latest.pt")
SIGMA = float(os.environ.get("SIGMA", "0.15")); R_MAX = float(os.environ.get("R_MAX", "0.15"))
N_GROUPS = int(os.environ.get("N_GROUPS", "8")); GROUP = int(os.environ.get("GROUP", "6"))
PPO_EPOCHS = int(os.environ.get("PPO_EPOCHS", "3")); MB = int(os.environ.get("MB", "64"))
LR = float(os.environ.get("LR", "1e-4")); CLIP = float(os.environ.get("CLIP", "0.2"))
KL_BETA = float(os.environ.get("KL_BETA", "0.1")); KL_STOP = float(os.environ.get("KL_STOP", "0.02"))
GAMMA_RTG = float(os.environ.get("GAMMA_RTG", "1.0")); WALL = float(os.environ.get("WALL_BUDGET", "6000"))
EP_CHUNKS = int(os.environ.get("EP_CHUNKS", "23")); N_EVAL_PERIODIC = int(os.environ.get("N_EVAL_PERIODIC", "8"))
RESET_SUITE_SEED = int(os.environ.get("RESET_SUITE_SEED", "20260817")); SEED = int(os.environ.get("SEED", "0"))
res = {"_status": "RUNNING", "algo": "grpo_residual", "config": {k: globals()[k] for k in
       ("SIGMA","R_MAX","N_GROUPS","GROUP","PPO_EPOCHS","MB","LR","CLIP","KL_BETA","KL_STOP","GAMMA_RTG","WALL","EP_CHUNKS","N_EVAL_PERIODIC","RESET_SUITE_SEED","SEED","BASE_CKPT")},
       "iterations": [], "evals": [], "best": None}
def emit(s="RUNNING"):
    res["_status"] = s
    with open(OUT, "w") as f: json.dump(res, f, indent=2, default=str)
t_start = time.time()
try:
    os.environ.pop("DISPLAY", None)
    sys.path.insert(0, "/home/jren313/research/starvla_rl/starVLA")
    import torch
    from omegaconf import OmegaConf
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    CKPT = "/home/jren313/research/starvla_rl/checkpoints/g1-longhorizon-oft-v1/final_model/pytorch_model.pt"
    CFGY = "/home/jren313/research/starvla_rl/checkpoints/g1-longhorizon-oft-v1/config.yaml"
    STATS = "/home/jren313/research/starvla_rl/checkpoints/g1-longhorizon-oft-v1/dataset_statistics.json"
    TASK = "pick up the piston with the right hand, inject it into the tube held by the left hand, then move it over the hole plate."
    RL = "/home/jren313/research/starvla_rl/RLinf/rlinf/envs/isaaclab/tasks/"
    import importlib.util as ilu
    def _load(m, p):
        sp = ilu.spec_from_file_location(m, p); mo = ilu.module_from_spec(sp)
        sys.modules[m] = mo; sp.loader.exec_module(mo); return mo
    Q99 = _load("g1n", RL + "g1_piston_norm.py").Q99ActionNormalizer
    Mapper = _load("g1a", RL + "g1_piston_action.py").G1PistonActionMapper
    HR = _load("g1h", RL + "g1_piston_hand_retarget.py")
    RW3 = _load("g1r3", RL + "g1_piston_reward_v3.py")
    RESP = _load("g1rp", RL + "g1_piston_residual_policy.py")
    GRPO = _load("g1g", RL + "g1_piston_grpo.py")
    RLSP = _load("g1s", RL + "g1_piston_rl_space.py")
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=True).app
    sys.path.append("/home/jren313/miniconda3/envs/isaac/lib/python3.11/site-packages")
    sys.path.insert(0, "/home/jren313/unitree_sim_isaaclab")
    import tasks, gymnasium as gym
    from isaaclab_tasks.utils import load_cfg_from_registry
    TID = "Isaac-PickPlace-Piston-G129-Inspire-Joint"
    ecfg = load_cfg_from_registry(TID, "env_cfg_entry_point")
    ecfg.seed = SEED; ecfg.scene.num_envs = 1
    ecfg.scene.left_wrist_camera = None; ecfg.scene.right_wrist_camera = None
    env = gym.make(TID, cfg=ecfg, render_mode="rgb_array").unwrapped
    sc = env.scene
    jn = list(sc["robot"].data.joint_names)
    mapper = Mapper(jn); retarget = HR.InspireHandRetargeter(jn)
    reward_fn = RW3.PistonTaskRewardV3(sc, jn)
    res["sim_ok"] = True; emit()
    mcfg = OmegaConf.load(CFGY)
    from starVLA.model.framework.base_framework import build_framework
    from deployment.model_server.tools.image_tools import to_pil_preserve
    from starVLA.training.trainer_utils.trainer_tools import resize_images
    model = build_framework(mcfg)
    model.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=False), strict=False)
    model = model.to("cuda"); DEV = "cuda"
    nrm = Q99.from_dataset_statistics(STATS)
    H = int(model.action_horizon); HID = int(model.qwen_vl_interface.model.config.hidden_size)
    sys.path.insert(0, "/home/jren313/research/starvla_rl/RLinf")
    from rlinf.envs.isaaclab.tasks.g1_piston_reset import apply_reset_condition, build_reset_suite, suite_manifest
    ACTION_LOW, ACTION_HIGH = -2.2, 2.2
    ACT_MASK = RLSP.build_active_mask().to(DEV); FROZEN_V = RLSP.load_frozen_values(STATS).to(DEV).float()
    # FROZEN base head = the BC policy; only the residual trains.
    bc = torch.load(BASE_CKPT, map_location="cpu", weights_only=False)
    model.action_model.load_state_dict(bc["action_model"])
    for p in model.parameters(): p.requires_grad_(False)
    model.eval()
    residual = RESP.ResidualPolicy(feat_dim=HID, hidden=(512, 512), r_max=R_MAX, horizon=H, dims=30, device=DEV)
    # Continuation: start from a previous run's residual (its base head is the same BC head).
    INIT_RESIDUAL = os.environ.get("INIT_RESIDUAL", "")
    if INIT_RESIDUAL:
        _prev = torch.load(INIT_RESIDUAL, map_location="cpu", weights_only=False)
        residual.load_state_dict(_prev["residual"])
        res["config"]["init_residual"] = INIT_RESIDUAL
        res["config"]["init_residual_iteration"] = int(_prev.get("iteration", -1))
    K = residual.n_basis
    opt = torch.optim.Adam(residual.parameters(), lr=LR)
    TRAIN_CONDITIONS, EVAL_CONDITIONS = build_reset_suite(n_train=400, n_eval=50, seed=RESET_SUITE_SEED)
    res["reset_suite"] = suite_manifest(TRAIN_CONDITIONS, EVAL_CONDITIONS)
    res["setup"] = {"H": H, "HID": HID, "K": K, "n_active": int(ACT_MASK.sum()),
                    "residual_params": int(sum(p.numel() for p in residual.parameters()))}
    emit()

    # ---------------- helpers (identical to train_sac.py where they overlap) -----------
    def vlm_encode(img_np):
        with torch.no_grad():
            imgs = [to_pil_preserve([img_np])]
            size = getattr(model.config.datasets.vla_data, "obs_image_size", None)
            if size: imgs = resize_images(imgs, target_size=size)
            toks = model.action_token * model.chunk_len
            instr = TASK + f" Please predict the next {model.chunk_len} robot actions: <action>{toks}<action>."
            qi = model.qwen_vl_interface.build_qwenvl_inputs(images=imgs, instructions=[instr])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                qo = model.qwen_vl_interface(**qi, output_attentions=False, output_hidden_states=True, return_dict=True)
                last_hidden = qo.hidden_states[-1]
            aq = model._gather_action_token_embeddings(last_hidden, qi.get("input_ids", None), action_token_id=model.action_token_id)
            return aq.detach().float()
    def base_action(aq):
        """Frozen head -> deterministic executed chunk [1,H,30] (squash + frozen dims)."""
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float32):
            mean = model.action_model.predict_action(aq).float()
        b, c, d = mean.shape
        a = torch.tanh(mean.reshape(b * c, d)) * (ACTION_HIGH - ACTION_LOW) / 2 + (ACTION_HIGH + ACTION_LOW) / 2
        return torch.where(ACT_MASK, a, FROZEN_V.expand_as(a)).reshape(b, c, d)
    def act_to_command(norm_action):
        phys = nrm.denormalize(norm_action.detach().cpu())
        cmd = mapper.map(phys).to(env.device)
        return retarget.apply(cmd, phys.to(env.device))
    def run_chunk(norm_action):
        cmd = act_to_command(norm_action); total_r = 0.0; done = False; info = {}
        for t in range(H):
            a = cmd[t].unsqueeze(0)
            for _hold in range(2):
                _, _, te, tr, _ = env.step(a)
                if bool(te[0]) or bool(tr[0]): done = True
            r, info = reward_fn.step(); total_r += r
            if done: break
        return total_r, done, info
    def get_img():
        return sc["front_camera"].data.output["rgb"][0].cpu().numpy().copy()

    def rollout(cond, stochastic):
        """One episode. Returns dict with per-chunk storage (for PPO) and outcome."""
        env.reset(seed=0); apply_reset_condition(env, cond); reward_fn.reset()
        bar0 = sc["object"].data.body_pos_w[0, 1].cpu().numpy().copy()
        aqs, bases, cs, rews = [], [], [], []; stages = {}; maxlift = 0.0; maxpress = 0.0
        for _ in range(EP_CHUNKS):
            aq = vlm_encode(get_img()); a_base = base_action(aq)
            with torch.no_grad():
                c_mean = residual.coeff_mean(aq, a_base)
                c = c_mean + SIGMA * torch.randn_like(c_mean) if stochastic else c_mean
                a = residual.compose(a_base, c, ACT_MASK)
                a = torch.where(ACT_MASK, a, FROZEN_V.expand_as(a))
            r, done, info = run_chunk(a[0])
            aqs.append(aq[0].to("cpu", dtype=torch.float16)); bases.append(a_base[0].to("cpu", dtype=torch.float16))
            cs.append(c[0].cpu()); rews.append(float(r))
            for k, v in info.get("stages", {}).items(): stages[k] = stages.get(k, False) or v
            bar = sc["object"].data.body_pos_w[0, 1].cpu().numpy()
            maxlift = max(maxlift, float(bar[2] - bar0[2])); maxpress = max(maxpress, float(info.get("max_press_m", 0.0)))
            if done: break
        bar = sc["object"].data.body_pos_w[0, 1].cpu().numpy()
        return {"aq": aqs, "base": bases, "c": cs, "rew": rews, "ret": float(sum(rews)),
                "stages": {k: bool(v) for k, v in stages.items()}, "max_lift_m": round(maxlift, 4),
                "max_press_m": round(maxpress, 5), "disp_m": round(float(np.linalg.norm(bar - bar0)), 4),
                "condition": cond.index}

    def save_ckpt(name, it, n_updates):
        ck = {"action_model": model.action_model.state_dict(), "residual": residual.state_dict(),
              "residual_cfg": {"feat_dim": HID, "hidden": [512, 512], "r_max": R_MAX},
              "actor_logstd": torch.full((30,), math.log(SIGMA)), "env_steps": it * N_GROUPS * GROUP * EP_CHUNKS * H * 2,
              "grad_updates": n_updates, "iteration": it, "algo": "grpo_residual"}
        torch.save(ck, f"{RUN_DIR}/{name}")

    def evaluate(it):
        rows = [rollout(cond, stochastic=False) for cond in EVAL_CONDITIONS[:N_EVAL_PERIODIC]]
        n = len(rows)
        out = {"iteration": it, "n": n, "mean_return": round(sum(r["ret"] for r in rows) / n, 3)}
        for st in ("reach", "grasp", "lift", "plate", "press", "dispense", "success"):
            out[st + "_rate"] = round(sum(1 for r in rows if r["stages"].get(st)) / n, 3)
        out["mean_max_lift_m"] = round(sum(r["max_lift_m"] for r in rows) / n, 4)
        out["max_press_m_any"] = round(max(r["max_press_m"] for r in rows), 5)
        res["evals"].append(out); emit(); return out

    # ---------------- iteration 0: the base policy (zero residual), for the record -----
    n_updates = 0; save_ckpt("grpo_ckpt_iter0.pt", 0, 0)
    ev0 = evaluate(0); best = {"iteration": 0, "mean_return": ev0["mean_return"], "grasp": ev0["grasp_rate"], "lift": ev0["lift_rate"]}
    save_ckpt("grpo_ckpt_best.pt", 0, 0); res["best"] = best; emit()

    it = 0; cond_ptr = 0
    while time.time() - t_start < WALL:
        it += 1; t_it = time.time()
        # ---- collect ----
        groups = []; online = []
        for g in range(N_GROUPS):
            cond = TRAIN_CONDITIONS[cond_ptr % len(TRAIN_CONDITIONS)]; cond_ptr += 1
            eps = [rollout(cond, stochastic=True) for _ in range(GROUP)]
            groups.append(eps); online.extend(eps)
        n_eps = len(online)
        coll = {"n_episodes": n_eps, "mean_return": round(sum(e["ret"] for e in online) / n_eps, 3),
                "return_std_within_group_mean": round(float(np.mean([np.std([e["ret"] for e in g]) for g in groups])), 3),
                **{st + "_rate": round(sum(1 for e in online if e["stages"].get(st)) / n_eps, 3) for st in ("grasp", "lift", "plate", "press", "dispense")},
                "max_press_m_any": round(max(e["max_press_m"] for e in online), 5), "collect_s": round(time.time() - t_it, 1)}
        # ---- advantages (GRPO) ----
        AQ, BS, CC, ADV = [], [], [], []
        for eps in groups:
            rtgs = [GRPO.returns_to_go(torch.tensor(e["rew"]), GAMMA_RTG) for e in eps]
            advs = GRPO.group_advantages(rtgs)
            for e, a in zip(eps, advs):
                AQ.extend(e["aq"]); BS.extend(e["base"]); CC.extend(e["c"]); ADV.extend(a.tolist())
        N = len(CC); ADV_t = torch.tensor(ADV, dtype=torch.float32)
        # ---- old log-probs under the pre-update residual ----
        logp_old = torch.zeros(N)
        with torch.no_grad():
            for i in range(0, N, MB):
                aq = torch.stack(AQ[i:i + MB]).to(DEV).float(); bs = torch.stack(BS[i:i + MB]).to(DEV).float()
                c = torch.stack(CC[i:i + MB]).to(DEV)
                mu = residual.coeff_mean(aq, bs)
                logp_old[i:i + MB] = GRPO.gaussian_logp_mean(c, mu, SIGMA, ACT_MASK).cpu()
        # ---- PPO epochs ----
        ppo = {"epochs_run": 0, "approx_kl": [], "clipfrac": [], "loss": [], "kl_base": [], "early_stop": False}
        for ep in range(PPO_EPOCHS):
            perm = torch.randperm(N); kls = []
            for i in range(0, N, MB):
                idx = perm[i:i + MB]
                aq = torch.stack([AQ[j] for j in idx]).to(DEV).float(); bs = torch.stack([BS[j] for j in idx]).to(DEV).float()
                c = torch.stack([CC[j] for j in idx]).to(DEV); adv = ADV_t[idx].to(DEV); lpo = logp_old[idx].to(DEV)
                mu = residual.coeff_mean(aq, bs)
                lpn = GRPO.gaussian_logp_mean(c, mu, SIGMA, ACT_MASK)
                loss_pg, info = GRPO.ppo_clipped_loss(lpn, lpo, adv, CLIP)
                kl_b = GRPO.kl_to_base_mean(mu, SIGMA, ACT_MASK).mean()
                loss = loss_pg + KL_BETA * kl_b
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(residual.parameters(), 1.0); opt.step(); n_updates += 1
                kls.append(info["approx_kl"]); ppo["clipfrac"].append(info["clipfrac"]); ppo["loss"].append(float(loss)); ppo["kl_base"].append(float(kl_b))
            ppo["epochs_run"] += 1; ppo["approx_kl"].append(float(np.mean(kls)))
            if np.mean(kls) > KL_STOP: ppo["early_stop"] = True; break
        ppo = {k: (round(float(np.mean(v)), 5) if isinstance(v, list) and v and k != "approx_kl" else v) for k, v in ppo.items()}
        ppo["approx_kl"] = [round(x, 5) for x in ppo["approx_kl"]]
        # ---- evaluate, checkpoint, best ----
        ev = evaluate(it)
        save_ckpt("grpo_ckpt_latest.pt", it, n_updates); save_ckpt(f"grpo_ckpt_iter{it}.pt", it, n_updates)
        if ev["mean_return"] > best["mean_return"]:
            best = {"iteration": it, "mean_return": ev["mean_return"], "grasp": ev["grasp_rate"], "lift": ev["lift_rate"]}
            save_ckpt("grpo_ckpt_best.pt", it, n_updates)
        res["best"] = best
        res["iterations"].append({"iteration": it, "collect": coll, "ppo": ppo, "n_chunks": N, "n_updates": n_updates,
                                  "eval": {k: ev[k] for k in ("mean_return", "grasp_rate", "lift_rate", "plate_rate", "press_rate")},
                                  "residual_coeff_abs_mean": round(float(torch.stack(CC).abs().mean()), 4),
                                  "wall_s": round(time.time() - t_start, 1)})
        emit()
    res["totals"] = {"iterations": it, "grad_updates": n_updates, "wall_clock_s": round(time.time() - t_start, 1),
                     "peak_vram_mib": int(torch.cuda.max_memory_allocated() / 1024 ** 2)}
    emit("OK"); os._exit(0)
except Exception as e:
    res["error"] = f"{type(e).__name__}: {e}"; res["tb"] = traceback.format_exc(); emit("FAIL"); os._exit(1)
