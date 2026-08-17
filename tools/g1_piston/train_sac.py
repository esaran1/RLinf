"""Single-process SAC / RLPD trainer for StarVLA QwenOFT on the IsaacLab G1 piston task.

Why single-process: Isaac Sim must own the process (it boots a Kit kernel and its USD
stage is not fork-safe), and the whole experiment fits on one 16 GB card (measured 9.46
GB peak with 6.9 GB headroom). RLinf's SAC worker assumes Ray-distributed env/actor
nodes, so this reuses its *math* components (MultiQHead, EntropyTemperature) and the
project's own StarVLA SAC forward semantics, rather than its multi-node plumbing.

ALGO=sac   -> online replay only
ALGO=rlpd  -> online replay + frozen verified-executable demo buffer, mixed 50/50

Everything else is identical between the two by construction: same SFT init, same reset
distribution, same reward, same 20-D action mask, same eval seeds, same interaction
budget. Nothing task-side is tuned per algorithm.
"""
import os, sys, json, time, traceback, random
from collections import deque
import numpy as np

OUT = os.environ["OUTF"]
ALGO = os.environ.get("ALGO", "sac").lower()          # sac | rlpd | sft_eval
RUN_DIR = os.environ["RUN_DIR"]
MAX_ENV_STEPS = int(os.environ.get("MAX_ENV_STEPS", "60000"))   # online env steps
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "6000"))          # env steps between evals
EVAL_SEEDS = [int(s) for s in os.environ.get("EVAL_SEEDS", "0,1,2,3,4").split(",")]
UTD = float(os.environ.get("UTD", "0.5"))             # gradient updates per env decision
BATCH = int(os.environ.get("BATCH", "8"))
DEMO_FRAC = float(os.environ.get("DEMO_FRAC", "0.5")) # RLPD offline mix
SEED = int(os.environ.get("SEED", "0"))
EP_CHUNKS = int(os.environ.get("EP_CHUNKS", "23"))    # 23 chunks x 30 x 2 = 1380 env steps
DEMO_DIR = "/home/jren313/research/starvla_rl/demo_buffer"

os.makedirs(RUN_DIR, exist_ok=True)
res = {
    "algo": ALGO, "seed": SEED, "config": {
        "max_env_steps": MAX_ENV_STEPS, "eval_every": EVAL_EVERY, "eval_seeds": EVAL_SEEDS,
        "utd": UTD, "batch": BATCH, "demo_frac": DEMO_FRAC if ALGO == "rlpd" else 0.0,
        "ep_chunks": EP_CHUNKS,
    },
    "train_log": [], "evals": [], "episodes": [],
}

def emit(s="RUNNING"):
    res["_status"] = s
    with open(OUT, "w") as f: json.dump(res, f, indent=2, default=str)

t_start = time.time()
try:
    os.environ.pop("DISPLAY", None)
    sys.path.insert(0, "/home/jren313/research/starvla_rl/starVLA")
    import torch
    import torch.nn.functional as F
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
    RW = _load("g1r", RL + "g1_piston_reward.py")
    RLSP = _load("g1s", RL + "g1_piston_rl_space.py")

    # ---------------- simulator ----------------
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
    reward_fn = RW.PistonTaskReward(sc, jn)
    res["sim_ok"] = True; emit()

    # ---------------- policy ----------------
    mcfg = OmegaConf.load(CFGY)
    from starVLA.model.framework.base_framework import build_framework
    from deployment.model_server.tools.image_tools import to_pil_preserve
    from starVLA.training.trainer_utils.trainer_tools import resize_images
    model = build_framework(mcfg)
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(sd, strict=False)
    model = model.to("cuda")
    nrm = Q99.from_dataset_statistics(STATS)
    H = int(model.action_horizon)
    HID = int(model.qwen_vl_interface.model.config.hidden_size)
    DEV = "cuda"

    sys.path.insert(0, "/home/jren313/research/starvla_rl/RLinf")
    from rlinf.models.embodiment.modules.q_head import MultiQHead
    from rlinf.models.embodiment.modules.entropy_tunning import EntropyTemperature
    from rlinf.models.embodiment.modules.gaussian_policy import SquashedNormal

    ACTION_LOW, ACTION_HIGH = -2.2, 2.2
    ACT_MASK = RLSP.build_active_mask().to(DEV)                       # [30] bool
    FROZEN_V = RLSP.load_frozen_values(STATS).to(DEV).float()         # [30]
    N_ACTIVE = int(ACT_MASK.sum())

    # freeze VLM, train OFT head only
    for p in model.parameters(): p.requires_grad_(False)
    for p in model.action_model.parameters(): p.requires_grad_(True)
    vlm_ref = {n: p.detach().clone() for n, p in list(
        model.qwen_vl_interface.named_parameters())[:5]}   # spot-check frozen-ness
    oft_ref = {n: p.detach().clone() for n, p in list(
        model.action_model.named_parameters())[:5]}

    # exploration log-std (per active dim), learned
    actor_logstd = torch.nn.Parameter(torch.full((30,), -1.0, device=DEV))

    critic = MultiQHead(HID, 30 * H, [256, 256], num_q_heads=2).to(DEV)
    target = MultiQHead(HID, 30 * H, [256, 256], num_q_heads=2).to(DEV)
    target.load_state_dict(critic.state_dict())
    for p in target.parameters(): p.requires_grad_(False)

    ent = EntropyTemperature(initial_alpha=0.01, alpha_type="softplus", device=DEV).to(DEV)
    TARGET_ENTROPY = RLSP.default_target_entropy()      # -20.0, active dims only

    opt_actor = torch.optim.Adam(
        list(model.action_model.parameters()) + [actor_logstd], lr=1e-5)
    opt_critic = torch.optim.Adam(critic.parameters(), lr=3e-4)
    opt_alpha = torch.optim.Adam(ent.parameters(), lr=3e-4)

    GAMMA = 0.99 ** H       # chunk-level discount (one transition = H actions)
    TAU = 0.005
    res["setup"] = {
        "hidden": HID, "action_horizon": H, "n_active_dims": N_ACTIVE,
        "target_entropy": TARGET_ENTROPY, "gamma_chunk": round(GAMMA, 5),
        "oft_params": int(sum(p.numel() for p in model.action_model.parameters())),
        "critic_params": int(sum(p.numel() for p in critic.parameters())),
    }
    emit()

    # ---------------- helpers ----------------
    def vlm_encode(img_np):
        """Run the FROZEN backbone once. Returns (action_queries [1,H,HID], pooled [1,HID]).

        ``QwenOFT.predict_action`` is decorated ``@torch.inference_mode()`` and detaches
        to numpy, so it cannot carry an actor gradient. Splitting the forward here lets
        the VLM stay frozen (and run once per observation) while the OFT head is re-run
        with grad on each update -- which is also far cheaper than a VLM pass per
        gradient step.
        """
        with torch.no_grad():
            imgs = [to_pil_preserve([img_np])]
            instr = TASK
            size = getattr(model.config.datasets.vla_data, "obs_image_size", None)
            if size:
                imgs = resize_images(imgs, target_size=size)
            toks = model.action_token * model.chunk_len
            instr = instr + (f" Please predict the next {model.chunk_len} robot actions:"
                             f" <action>{toks}<action>.")
            qi = model.qwen_vl_interface.build_qwenvl_inputs(images=imgs, instructions=[instr])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                qo = model.qwen_vl_interface(**qi, output_attentions=False,
                                             output_hidden_states=True, return_dict=True)
                last_hidden = qo.hidden_states[-1]
            aq = model._gather_action_token_embeddings(
                last_hidden, qi.get("input_ids", None),
                action_token_id=model.action_token_id)          # [1, chunk_len, HID]
            am = qi.get("attention_mask", None)
            if am is not None:
                m = am.to(last_hidden.dtype).unsqueeze(-1)
                pooled = (last_hidden * m).sum(1) / m.sum(1).clamp(min=1e-6)
            else:
                pooled = last_hidden.mean(1)
        return aq.detach().float(), pooled.detach().float()

    def head_mean(action_queries):
        """Differentiable OFT head forward: action_queries -> mean action [B,H,30]."""
        with torch.autocast("cuda", dtype=torch.float32):
            return model.action_model.predict_action(action_queries)

    def vlm_feature_and_mean(img_np):
        """(pooled state feature [1,HID], mean action [1,H,30]) -- no grad, for rollout."""
        aq, pooled = vlm_encode(img_np)
        with torch.no_grad():
            mean = head_mean(aq).float()
        return pooled, mean, aq

    def sample_action(mean, logstd, deterministic=False):
        """SquashedNormal sample with per-dim masking. mean [B,H,30]."""
        b, c, d = mean.shape
        flat = mean.reshape(b * c, d)
        if deterministic:
            scale = (ACTION_HIGH - ACTION_LOW) / 2.0
            shift = (ACTION_HIGH + ACTION_LOW) / 2.0
            a = torch.tanh(flat) * scale + shift
            lp = torch.zeros(b * c, device=flat.device)
        else:
            std = torch.exp(logstd).view(1, -1).expand_as(flat)
            dist = SquashedNormal(flat, std, low=ACTION_LOW, high=ACTION_HIGH)
            a = dist.rsample()
            lp = masked_logprob(dist, a)
        a = torch.where(ACT_MASK, a, FROZEN_V.expand_as(a))
        return a.reshape(b, c, d), lp.reshape(b, c)

    def masked_logprob(dist, flat_sample):
        base = dist.base_dist.base_dist
        parts = []
        for t in dist.transforms:
            parts.extend(getattr(t, "parts", [t]))
        x = flat_sample
        for t in reversed(parts):
            x = t.inv(x)
        per_dim = base.log_prob(x)
        y = x
        for t in parts:
            y2 = t(y)
            nm = type(t).__name__
            if nm == "TanhTransform":
                per_dim = per_dim - torch.log(1 - y2.pow(2) + 1e-7)
            else:
                s = torch.as_tensor(t.scale, dtype=x.dtype, device=x.device)
                per_dim = per_dim - torch.log(s.abs()).expand_as(x)
            y = y2
        return (per_dim * ACT_MASK).sum(dim=-1)

    def act_to_command(norm_action):
        """[H,30] normalized -> [H,53] retargeted simulator command."""
        phys = nrm.denormalize(norm_action.detach().cpu())
        cmd = mapper.map(phys).to(env.device)
        return retarget.apply(cmd, phys.to(env.device))

    def run_chunk(norm_action):
        """Execute one H-action chunk with 2x hold. Returns (reward, done, info)."""
        cmd = act_to_command(norm_action)
        total_r = 0.0; done = False; info = {}
        for t in range(H):
            a = cmd[t].unsqueeze(0)
            for _hold in range(2):
                _, _, te, tr, _ = env.step(a)
                if bool(te[0]) or bool(tr[0]): done = True
            r, info = reward_fn.step()
            total_r += r
            if done: break
        return total_r, done, info

    def get_img():
        return sc["front_camera"].data.output["rgb"][0].cpu().numpy().copy()

    # ---------------- evaluation suite (FIXED, reused throughout) ----------------
    def evaluate(tag, env_steps, save_video=False):
        """Deterministic policy on the fixed eval seeds. Behavior metrics only."""
        rows = []
        for s in EVAL_SEEDS:
            env.reset(seed=s); reward_fn.reset()
            bar0 = sc["object"].data.body_pos_w[0, 1].cpu().numpy().copy()
            ret = 0.0; maxlift = 0.0; stages = {}; frames = []
            for c in range(EP_CHUNKS):
                img = get_img()
                if save_video and s == EVAL_SEEDS[0]: frames.append(img)
                _, mean, _ = vlm_feature_and_mean(img)
                a, _ = sample_action(mean, actor_logstd, deterministic=True)
                r, done, info = run_chunk(a[0])
                ret += r
                for k, v in info.get("stages", {}).items():
                    stages[k] = stages.get(k, False) or v
                bar = sc["object"].data.body_pos_w[0, 1].cpu().numpy()
                maxlift = max(maxlift, float(bar[2] - bar0[2]))
                if done: break
            bar = sc["object"].data.body_pos_w[0, 1].cpu().numpy()
            rows.append({
                "seed": s, "return": round(float(ret), 3),
                "disp_m": round(float(np.linalg.norm(bar - bar0)), 4),
                "max_lift_m": round(maxlift, 4),
                "stages": {k: bool(v) for k, v in stages.items()},
            })
            if save_video and s == EVAL_SEEDS[0] and frames:
                try:
                    import imageio.v2 as imageio
                    imageio.mimsave(f"{RUN_DIR}/{ALGO}_step{env_steps}_seed{s}.mp4",
                                    frames, fps=8)
                except Exception:
                    pass
        n = len(rows)
        def rate(k): return round(sum(1 for r in rows if r["stages"].get(k)) / n, 3)
        ev = {
            "tag": tag, "env_steps": env_steps,
            "wall_clock_s": round(time.time() - t_start, 1),
            "full_success_rate": rate("success"), "reach_rate": rate("reach"),
            "grasp_rate": rate("grasp"), "lift_rate": rate("lift"),
            "plate_rate": rate("plate"), "tube_rate": rate("tube"),
            "mean_return": round(float(np.mean([r["return"] for r in rows])), 3),
            "mean_disp_m": round(float(np.mean([r["disp_m"] for r in rows])), 4),
            "mean_max_lift_m": round(float(np.mean([r["max_lift_m"] for r in rows])), 4),
            "per_seed": rows,
        }
        res["evals"].append(ev); emit()
        return ev

    # ---------------- SFT baseline: evaluate once, no training ----------------
    if ALGO == "sft_eval":
        evaluate("sft_baseline", 0, save_video=True)
        res["wall_clock_s"] = round(time.time() - t_start, 1)
        emit("OK"); os._exit(0)

    # ---------------- replay buffers ----------------
    online = deque(maxlen=20000)     # (feat, mean_action, action, reward, next_feat, done)
    demo = []
    if ALGO == "rlpd":
        for fn in sorted(os.listdir(DEMO_DIR)):
            if not fn.endswith(".npz"): continue
            z = np.load(os.path.join(DEMO_DIR, fn))
            imgs, acts, rews = z["images"], z["actions"], z["rewards"]
            for i in range(len(acts) - 1):
                demo.append((imgs[i], acts[i], float(rews[i]), imgs[i + 1], False))
        res["demo_transitions"] = len(demo)
        res["demo_episodes"] = len([f for f in os.listdir(DEMO_DIR) if f.endswith(".npz")])
        emit()

    demo_feat_cache = {}
    def demo_batch(k):
        """Sample k demo transitions, computing (and caching) VLM encodings lazily.

        The demo images come from simulator replay under the frozen retargeter, so their
        VLM encoding is fixed; caching makes RLPD's offline half nearly free after the
        first pass.
        """
        out = []
        for idx in np.random.randint(0, len(demo), size=k):
            img, act, r, nimg, d = demo[idx]
            if idx not in demo_feat_cache:
                aq, f = vlm_encode(img)
                naq, nf = vlm_encode(nimg)
                demo_feat_cache[idx] = (f.squeeze(0).cpu(), aq.squeeze(0).cpu(),
                                        nf.squeeze(0).cpu(), naq.squeeze(0).cpu())
            f, aq, nf, naq = demo_feat_cache[idx]
            a = torch.tensor(act, dtype=torch.float32)
            out.append((f, aq, a, r, nf, naq, d))
        return out

    def collate(items):
        """items: (pooled, action_queries, action, reward, next_pooled, next_aq, done)"""
        feats = torch.stack([x[0] for x in items]).to(DEV)
        aqs = torch.stack([x[1] for x in items]).to(DEV)
        acts = torch.stack([x[2] for x in items]).to(DEV)
        rews = torch.tensor([x[3] for x in items], dtype=torch.float32, device=DEV).unsqueeze(-1)
        nfeats = torch.stack([x[4] for x in items]).to(DEV)
        naqs = torch.stack([x[5] for x in items]).to(DEV)
        dones = torch.tensor([x[6] for x in items], dtype=torch.bool, device=DEV).unsqueeze(-1)
        return feats, aqs, acts, rews, nfeats, naqs, dones

    # ---------------- gradient update ----------------
    def sac_update(batch_items):
        feats, aqs, acts, rews, nfeats, naqs, dones = collate(batch_items)
        B = feats.shape[0]

        # --- critic ---
        with torch.no_grad():
            # next action from the CURRENT policy at the sampled next states
            nmean = head_mean(naqs).float()
            na, nlp = sample_action(nmean, actor_logstd)
            qn = target(nfeats, na.reshape(B, -1))
            qmin = qn.min(dim=1, keepdim=True)[0]
            alpha = ent.compute_alpha().detach()
            qmin = qmin - alpha * nlp.sum(dim=-1, keepdim=True)
            tq = rews + (~dones) * GAMMA * qmin
        q = critic(feats, acts.reshape(B, -1))
        closs = F.mse_loss(q, tq.expand_as(q))
        opt_critic.zero_grad(); closs.backward()
        cgn = torch.nn.utils.clip_grad_norm_(critic.parameters(), 10.0)
        opt_critic.step()

        # --- actor: gradient flows head_mean -> rsample -> critic ---
        mean_b = head_mean(aqs).float()                 # differentiable in OFT head
        a, lp = sample_action(mean_b, actor_logstd)
        qpi = critic(feats, a.reshape(B, -1)).min(dim=1, keepdim=True)[0]
        alpha = ent.compute_alpha().detach()
        aloss = (alpha * lp.sum(dim=-1, keepdim=True) - qpi).mean()
        opt_actor.zero_grad(); aloss.backward()
        agn = torch.nn.utils.clip_grad_norm_(
            list(model.action_model.parameters()) + [actor_logstd], 10.0)
        opt_actor.step()

        # --- alpha ---
        alpha_v = ent.compute_alpha()
        alloss = -alpha_v * (lp.sum(dim=-1).mean().detach() + TARGET_ENTROPY)
        opt_alpha.zero_grad(); alloss.backward(); opt_alpha.step()

        # --- target soft update ---
        with torch.no_grad():
            for tp, op in zip(target.parameters(), critic.parameters()):
                tp.data.mul_(1 - TAU).add_(op.data, alpha=TAU)

        return {
            "critic_loss": float(closs.item()), "actor_loss": float(aloss.item()),
            "alpha_loss": float(alloss.item()), "alpha": float(alpha_v.item()),
            "q_mean": float(q.mean().item()), "target_q_mean": float(tq.mean().item()),
            "logprob": float(lp.sum(dim=-1).mean().item()),
            "critic_gn": float(cgn), "actor_gn": float(agn),
        }

    # ---------------- training loop ----------------
    env_steps = 0; grad_updates = 0; episode = 0
    n_online_samples = 0; n_demo_samples = 0
    next_eval = 0

    evaluate("init", 0, save_video=True)   # step-0 behavior, same suite

    while env_steps < MAX_ENV_STEPS:
        ep_seed = 1000 + episode          # training resets: distinct from eval seeds
        env.reset(seed=ep_seed); reward_fn.reset()
        bar0 = sc["object"].data.body_pos_w[0, 1].cpu().numpy().copy()
        ep_ret = 0.0; ep_stages = {}; maxlift = 0.0
        img = get_img()
        feat, mean, aq = vlm_feature_and_mean(img)

        for c in range(EP_CHUNKS):
            a, _ = sample_action(mean, actor_logstd)          # stochastic: explore
            r, done, info = run_chunk(a[0])
            env_steps += H * 2
            nimg = get_img()
            nfeat, nmean, naq = vlm_feature_and_mean(nimg)
            online.append((feat.squeeze(0).cpu(), aq.squeeze(0).cpu(),
                           a[0].detach().cpu(), float(r),
                           nfeat.squeeze(0).cpu(), naq.squeeze(0).cpu(), bool(done)))
            ep_ret += r
            for k, v in info.get("stages", {}).items():
                ep_stages[k] = ep_stages.get(k, False) or v
            bar = sc["object"].data.body_pos_w[0, 1].cpu().numpy()
            maxlift = max(maxlift, float(bar[2] - bar0[2]))
            feat, mean, aq = nfeat, nmean, naq

            # --- gradient updates (UTD ratio) ---
            if len(online) >= BATCH:
                n_up = int(UTD) + (1 if random.random() < (UTD % 1) else 0)
                for _ in range(n_up):
                    if ALGO == "rlpd" and demo:
                        k_demo = int(BATCH * DEMO_FRAC)
                        k_on = BATCH - k_demo
                        items = random.sample(list(online), min(k_on, len(online)))
                        items = items + demo_batch(k_demo)
                        n_online_samples += min(k_on, len(online)); n_demo_samples += k_demo
                    else:
                        items = random.sample(list(online), BATCH)
                        n_online_samples += BATCH
                    m = sac_update(items)
                    grad_updates += 1
                    if grad_updates % 25 == 0:
                        m.update({"env_steps": env_steps, "grad_updates": grad_updates,
                                  "buffer": len(online), "n_online_samples": n_online_samples,
                                  "n_demo_samples": n_demo_samples,
                                  "demo_fraction": round(n_demo_samples /
                                      max(1, n_online_samples + n_demo_samples), 3)})
                        res["train_log"].append(m); emit()
            if done: break

        bar = sc["object"].data.body_pos_w[0, 1].cpu().numpy()
        res["episodes"].append({
            "episode": episode, "seed": ep_seed, "env_steps": env_steps,
            "return": round(float(ep_ret), 3),
            "disp_m": round(float(np.linalg.norm(bar - bar0)), 4),
            "max_lift_m": round(maxlift, 4),
            "stages": {k: bool(v) for k, v in ep_stages.items()},
        })
        episode += 1; emit()

        if env_steps >= next_eval:
            evaluate(f"periodic", env_steps, save_video=True)
            next_eval += EVAL_EVERY
            ck = {
                "action_model": model.action_model.state_dict(),
                "critic": critic.state_dict(), "target": target.state_dict(),
                "actor_logstd": actor_logstd.detach().cpu(),
                "alpha": ent.state_dict(),
                "env_steps": env_steps, "grad_updates": grad_updates,
            }
            torch.save(ck, f"{RUN_DIR}/{ALGO}_ckpt_latest.pt")
            res["last_checkpoint"] = f"{RUN_DIR}/{ALGO}_ckpt_latest.pt"
            # frozen-ness spot check
            with torch.no_grad():
                vlm_now = dict(list(model.qwen_vl_interface.named_parameters())[:5])
                oft_now = dict(list(model.action_model.named_parameters())[:5])
                res["vlm_unchanged"] = all(
                    torch.equal(vlm_ref[n].cpu(), vlm_now[n].detach().cpu()) for n in vlm_ref)
                res["oft_changed"] = any(
                    not torch.equal(oft_ref[n].cpu(), oft_now[n].detach().cpu()) for n in oft_ref)
            emit()

    evaluate("final", env_steps, save_video=True)
    res["totals"] = {
        "env_steps": env_steps, "grad_updates": grad_updates, "episodes": episode,
        "n_online_samples": n_online_samples, "n_demo_samples": n_demo_samples,
        "demo_fraction": round(n_demo_samples / max(1, n_online_samples + n_demo_samples), 3),
        "buffer_size": len(online),
        "wall_clock_s": round(time.time() - t_start, 1),
        "gpu_hours": round((time.time() - t_start) / 3600.0, 3),
        "peak_vram_mib": int(torch.cuda.max_memory_allocated() / 1024**2),
    }
    emit("OK"); os._exit(0)
except Exception as e:
    res["error"] = f"{type(e).__name__}: {e}"; res["tb"] = traceback.format_exc()
    res["wall_clock_s"] = round(time.time() - t_start, 1)
    emit("FAIL"); os._exit(1)
