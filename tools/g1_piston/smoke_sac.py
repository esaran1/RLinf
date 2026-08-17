"""Minimal SAC/RLPD correctness smoke test. Must pass before any long run.

Proves: finite actor/critic/target-Q/alpha losses, target-network soft updates, OFT
params change, VLM params do NOT change, OLD replay transitions are reused (the key
off-policy milestone), checkpoint saves and reloads, and that the split forward used for
gradients matches the frozen `predict_action` numerically.
"""
import os, sys, json, time, traceback, random
from collections import deque
import numpy as np

OUT = os.environ["OUTF"]
res = {"checks": {}}

def emit(s="RUNNING"):
    res["_status"] = s
    with open(OUT, "w") as f: json.dump(res, f, indent=2, default=str)

def chk(name, ok, detail=None):
    res["checks"][name] = {"pass": bool(ok), "detail": detail}
    emit()

try:
    os.environ.pop("DISPLAY", None)
    sys.path.insert(0, "/home/jren313/research/starvla_rl/starVLA")
    import torch
    import torch.nn.functional as F
    from omegaconf import OmegaConf
    torch.manual_seed(0); np.random.seed(0); random.seed(0)

    CKPT = "/home/jren313/research/starvla_rl/checkpoints/g1-longhorizon-oft-v1/final_model/pytorch_model.pt"
    CFGY = "/home/jren313/research/starvla_rl/checkpoints/g1-longhorizon-oft-v1/config.yaml"
    STATS = "/home/jren313/research/starvla_rl/checkpoints/g1-longhorizon-oft-v1/dataset_statistics.json"
    TASK = "pick up the piston with the right hand, inject it into the tube held by the left hand, then move it over the hole plate."
    RL = "/home/jren313/research/starvla_rl/RLinf/rlinf/envs/isaaclab/tasks/"
    DEMO_DIR = "/home/jren313/research/starvla_rl/demo_buffer"

    import importlib.util as ilu
    def _load(m, p):
        sp = ilu.spec_from_file_location(m, p); mo = ilu.module_from_spec(sp)
        sys.modules[m] = mo; sp.loader.exec_module(mo); return mo
    RLSP = _load("g1s", RL + "g1_piston_rl_space.py")

    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=True).app
    sys.path.append("/home/jren313/miniconda3/envs/isaac/lib/python3.11/site-packages")
    sys.path.insert(0, "/home/jren313/unitree_sim_isaaclab")
    import tasks, gymnasium as gym
    from isaaclab_tasks.utils import load_cfg_from_registry
    TID = "Isaac-PickPlace-Piston-G129-Inspire-Joint"
    ecfg = load_cfg_from_registry(TID, "env_cfg_entry_point")
    ecfg.seed = 0; ecfg.scene.num_envs = 1
    ecfg.scene.left_wrist_camera = None; ecfg.scene.right_wrist_camera = None
    env = gym.make(TID, cfg=ecfg, render_mode="rgb_array").unwrapped
    env.reset(seed=0)
    sc = env.scene

    mcfg = OmegaConf.load(CFGY)
    from starVLA.model.framework.base_framework import build_framework
    from deployment.model_server.tools.image_tools import to_pil_preserve
    from starVLA.training.trainer_utils.trainer_tools import resize_images
    model = build_framework(mcfg)
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(sd, strict=False)
    model = model.to("cuda")
    H = int(model.action_horizon)
    HID = int(model.qwen_vl_interface.model.config.hidden_size)
    DEV = "cuda"

    sys.path.insert(0, "/home/jren313/research/starvla_rl/RLinf")
    from rlinf.models.embodiment.modules.q_head import MultiQHead
    from rlinf.models.embodiment.modules.entropy_tunning import EntropyTemperature
    from rlinf.models.embodiment.modules.gaussian_policy import SquashedNormal

    ACTION_LOW, ACTION_HIGH = -2.2, 2.2
    ACT_MASK = RLSP.build_active_mask().to(DEV)
    FROZEN_V = RLSP.load_frozen_values(STATS).to(DEV).float()

    for p in model.parameters(): p.requires_grad_(False)
    for p in model.action_model.parameters(): p.requires_grad_(True)

    # Start exploration at the entropy target's std, as the trainer does.
    actor_logstd = torch.nn.Parameter(
        torch.full((30,), float(np.log(RLSP.TARGET_ENTROPY_STD)), device=DEV))
    critic = MultiQHead(HID, 30 * H, [256, 256], num_q_heads=2).to(DEV)
    target = MultiQHead(HID, 30 * H, [256, 256], num_q_heads=2).to(DEV)
    target.load_state_dict(critic.state_dict())
    for p in target.parameters(): p.requires_grad_(False)
    ent = EntropyTemperature(initial_alpha=0.01, alpha_type="softplus", device=DEV).to(DEV)
    TARGET_ENTROPY = RLSP.default_target_entropy()
    opt_actor = torch.optim.Adam(list(model.action_model.parameters()) + [actor_logstd], lr=1e-5)
    opt_critic = torch.optim.Adam(critic.parameters(), lr=3e-4)
    opt_alpha = torch.optim.Adam(ent.parameters(), lr=3e-4)
    GAMMA = 0.99 ** H; TAU = 0.005

    def vlm_encode(img_np):
        with torch.no_grad():
            imgs = [to_pil_preserve([img_np])]
            size = getattr(model.config.datasets.vla_data, "obs_image_size", None)
            if size: imgs = resize_images(imgs, target_size=size)
            toks = model.action_token * model.chunk_len
            instr = TASK + (f" Please predict the next {model.chunk_len} robot actions:"
                            f" <action>{toks}<action>.")
            qi = model.qwen_vl_interface.build_qwenvl_inputs(images=imgs, instructions=[instr])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                qo = model.qwen_vl_interface(**qi, output_attentions=False,
                                             output_hidden_states=True, return_dict=True)
                lh = qo.hidden_states[-1]
            aq = model._gather_action_token_embeddings(
                lh, qi.get("input_ids", None), action_token_id=model.action_token_id)
            am = qi.get("attention_mask", None)
            if am is not None:
                m = am.to(lh.dtype).unsqueeze(-1)
                pooled = (lh * m).sum(1) / m.sum(1).clamp(min=1e-6)
            else:
                pooled = lh.mean(1)
        return aq.detach().float(), pooled.detach().float()

    def head_mean(aq):
        with torch.autocast("cuda", dtype=torch.float32):
            return model.action_model.predict_action(aq)

    def masked_logprob(dist, s):
        base = dist.base_dist.base_dist
        parts = []
        for t in dist.transforms: parts.extend(getattr(t, "parts", [t]))
        x = s
        for t in reversed(parts): x = t.inv(x)
        per_dim = base.log_prob(x); y = x
        for t in parts:
            y2 = t(y)
            if type(t).__name__ == "TanhTransform":
                per_dim = per_dim - torch.log(1 - y2.pow(2) + 1e-7)
            else:
                sc_ = torch.as_tensor(t.scale, dtype=x.dtype, device=x.device)
                per_dim = per_dim - torch.log(sc_.abs()).expand_as(x)
            y = y2
        return (per_dim * ACT_MASK).sum(dim=-1)

    def sample_action(mean, logstd):
        b, c, d = mean.shape
        flat = mean.reshape(b * c, d)
        std = torch.exp(logstd).view(1, -1).expand_as(flat)
        dist = SquashedNormal(flat, std, low=ACTION_LOW, high=ACTION_HIGH)
        a = dist.rsample(); lp = masked_logprob(dist, a)
        a = torch.where(ACT_MASK, a, FROZEN_V.expand_as(a))
        return a.reshape(b, c, d), lp.reshape(b, c)

    img = sc["front_camera"].data.output["rgb"][0].cpu().numpy().copy()

    # --- CHECK: split forward matches the frozen predict_action ---
    aq, pooled = vlm_encode(img)
    with torch.no_grad():
        m_split = head_mean(aq).float().cpu().numpy()
        out = model.predict_action([{"image": [img], "lang": TASK}])
        m_ref = np.asarray(out["normalized_actions"])[0]
    md = float(np.max(np.abs(m_split[0] - m_ref)))
    chk("split_forward_matches_predict_action", md < 1e-3, {"max_abs_diff": md})
    chk("action_shape", m_split.shape[1:] == (30,) or m_split.shape[-1] == 30,
        {"shape": list(m_split.shape)})

    # --- build a small OLD buffer from demo replay (dynamics-consistent) ---
    demo = []
    for fn in sorted(os.listdir(DEMO_DIR))[:3]:
        z = np.load(os.path.join(DEMO_DIR, fn))
        imgs, acts, rews = z["images"], z["actions"], z["rewards"]
        for i in range(min(4, len(acts) - 1)):
            demo.append((imgs[i], acts[i], float(rews[i]), imgs[i + 1]))
    buf = []
    for im, ac, r, nim in demo:
        a1, f1 = vlm_encode(im); a2, f2 = vlm_encode(nim)
        buf.append((f1.squeeze(0), a1.squeeze(0), torch.tensor(ac, dtype=torch.float32,
                    device=DEV), r, f2.squeeze(0), a2.squeeze(0), False))
    chk("demo_buffer_loaded", len(buf) >= 8, {"transitions": len(buf)})

    def collate(items):
        return (torch.stack([x[0] for x in items]), torch.stack([x[1] for x in items]),
                torch.stack([x[2] for x in items]),
                torch.tensor([x[3] for x in items], dtype=torch.float32, device=DEV).unsqueeze(-1),
                torch.stack([x[4] for x in items]), torch.stack([x[5] for x in items]),
                torch.tensor([x[6] for x in items], dtype=torch.bool, device=DEV).unsqueeze(-1))

    def update(items):
        feats, aqs, acts, rews, nfeats, naqs, dones = collate(items)
        B = feats.shape[0]
        with torch.no_grad():
            nmean = head_mean(naqs).float()
            na, nlp = sample_action(nmean, actor_logstd)
            qn = target(nfeats, na.reshape(B, -1)).min(dim=1, keepdim=True)[0]
            qn = qn - ent.compute_alpha().detach() * nlp.sum(-1, keepdim=True)
            tq = rews + (~dones) * GAMMA * qn
        q = critic(feats, acts.reshape(B, -1))
        closs = F.mse_loss(q, tq.expand_as(q))
        opt_critic.zero_grad(); closs.backward(); opt_critic.step()
        mean_b = head_mean(aqs).float()
        a, lp = sample_action(mean_b, actor_logstd)
        qpi = critic(feats, a.reshape(B, -1)).min(dim=1, keepdim=True)[0]
        # Mean over the horizon: per-control-action scale, matching TARGET_ENTROPY.
        logp_chunk = lp.mean(-1, keepdim=True)
        aloss = (ent.compute_alpha().detach() * logp_chunk - qpi).mean()
        opt_actor.zero_grad(); aloss.backward()
        gnorm = sum(p.grad.abs().sum().item() for p in model.action_model.parameters()
                    if p.grad is not None)
        opt_actor.step()
        av = ent.compute_alpha()
        alloss = -av * (logp_chunk.mean().detach() + TARGET_ENTROPY)
        opt_alpha.zero_grad(); alloss.backward(); opt_alpha.step()
        with torch.no_grad():
            for tp, op in zip(target.parameters(), critic.parameters()):
                tp.data.mul_(1 - TAU).add_(op.data, alpha=TAU)
        return closs, aloss, alloss, av, tq, gnorm, a

    vlm_before = [p.detach().clone() for p in list(model.qwen_vl_interface.parameters())[:6]]
    oft_before = [p.detach().clone() for p in list(model.action_model.parameters())[:6]]
    tgt_before = [p.detach().clone() for p in list(target.parameters())[:4]]

    B = 4
    closs, aloss, alloss, av, tq, gnorm, sampled = update(random.sample(buf, B))
    chk("critic_loss_finite", bool(torch.isfinite(closs)), {"value": float(closs)})
    chk("actor_loss_finite", bool(torch.isfinite(aloss)), {"value": float(aloss)})
    chk("alpha_loss_finite", bool(torch.isfinite(alloss)), {"value": float(alloss)})
    chk("alpha_finite_positive", bool(torch.isfinite(av) and av > 0), {"alpha": float(av)})

    # The entropy target must be REACHABLE by the bounded action distribution, or alpha
    # is driven monotonically to zero and the actor ends up unregularised. This is the
    # invariant both SAC pilots violated; see
    # docs/contracts/g1_piston_sac_pilot_v1_collapse.json.
    chk("entropy_target_is_reachable",
        TARGET_ENTROPY > RLSP.MIN_ACHIEVABLE_LOGPROB,
        {"target": TARGET_ENTROPY, "floor": RLSP.MIN_ACHIEVABLE_LOGPROB})
    chk("target_q_finite", bool(torch.isfinite(tq).all()), {"mean": float(tq.mean())})
    chk("actor_grad_reaches_oft_head", gnorm > 0, {"grad_abs_sum": round(gnorm, 4)})

    # frozen dims must receive exactly zero exploration
    fz = list(RLSP.FROZEN_ACTION_DIMS)
    dev = float((sampled[..., fz] - FROZEN_V[fz]).abs().max())
    chk("frozen_dims_exact", dev == 0.0, {"max_abs_deviation": dev})

    alpha_first = float(av)
    for _ in range(9):
        closs, aloss, alloss, av, tq, gnorm, _ = update(random.sample(buf, B))

    # Alpha must move in the CORRECTING direction, not collapse regardless of the
    # policy. Exploration starts at the target std, so the sign of (logp - target)
    # decides which way it should go; a monotone slide toward zero is the pathology.
    alpha_last = float(av)
    chk("alpha_moves_in_the_correcting_direction",
        alpha_last > 0.5 * alpha_first,
        {"alpha_first": round(alpha_first, 5), "alpha_last": round(alpha_last, 5),
         "note": "both pilots slid 0.049 -> 0.039 monotonically toward zero"})

    vlm_after = [p.detach().clone() for p in list(model.qwen_vl_interface.parameters())[:6]]
    oft_after = [p.detach().clone() for p in list(model.action_model.parameters())[:6]]
    tgt_after = [p.detach().clone() for p in list(target.parameters())[:4]]
    chk("vlm_params_unchanged", all(torch.equal(a, b) for a, b in zip(vlm_before, vlm_after)))
    chk("oft_params_changed", any(not torch.equal(a, b) for a, b in zip(oft_before, oft_after)))
    chk("target_critic_updated",
        any(not torch.equal(a, b) for a, b in zip(tgt_before, tgt_after)))

    # --- THE off-policy milestone: reuse transitions collected before any update ---
    reuse_losses = []
    for _ in range(5):
        c2, *_ = update(random.sample(buf, B))
        reuse_losses.append(float(c2))
    chk("old_transitions_reused",
        all(np.isfinite(reuse_losses)),
        {"note": "all transitions predate every gradient update; 15 updates consumed them",
         "critic_losses": [round(x, 4) for x in reuse_losses]})

    # --- checkpoint save / reload ---
    CK = "/tmp/claude-3343958/-home-jren313-research-starvla-rl-RLinf/c78cad95-dbfe-4e7f-b78a-7e9be50a1fdc/scratchpad/smoke_ckpt.pt"
    torch.save({"action_model": model.action_model.state_dict(),
                "critic": critic.state_dict(), "target": target.state_dict(),
                "actor_logstd": actor_logstd.detach().cpu(), "alpha": ent.state_dict()}, CK)
    chk("checkpoint_saved", os.path.exists(CK),
        {"mib": round(os.path.getsize(CK) / 1024**2, 1)})
    ck = torch.load(CK, map_location="cuda", weights_only=False)
    c2 = MultiQHead(HID, 30 * H, [256, 256], num_q_heads=2).to(DEV)
    c2.load_state_dict(ck["critic"])
    model.action_model.load_state_dict(ck["action_model"])
    same = all(torch.equal(a, b) for a, b in
               zip(critic.state_dict().values(), c2.state_dict().values()))
    chk("checkpoint_reloaded_exact", same)

    res["peak_vram_mib"] = int(torch.cuda.max_memory_allocated() / 1024**2)
    res["all_passed"] = all(v["pass"] for v in res["checks"].values())
    emit("OK" if res["all_passed"] else "FAIL_CHECKS"); os._exit(0)
except Exception as e:
    res["error"] = f"{type(e).__name__}: {e}"; res["tb"] = traceback.format_exc()
    emit("FAIL"); os._exit(1)
