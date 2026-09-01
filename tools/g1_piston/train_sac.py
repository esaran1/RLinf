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
import os, sys, json, math, time, traceback, random
from collections import deque
import numpy as np

OUT = os.environ["OUTF"]
ALGO = os.environ.get("ALGO", "sac").lower()          # sac | rlpd | sft_eval
RUN_DIR = os.environ["RUN_DIR"]
MAX_ENV_STEPS = int(os.environ.get("MAX_ENV_STEPS", "60000"))   # online env steps
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "6000"))          # env steps between evals
#: Held-out initial conditions. The task has ONE deterministic reset, so evaluation
#: varies the initial state explicitly (g1_piston_reset).
#:
#: An episode costs ~58 s, so a 50-condition sweep is ~49 min. Periodic evals use the
#: first N_EVAL_PERIODIC conditions (a prefix of the same fixed, hashed suite, so the
#: curve is self-consistent) and the FINAL eval uses all 50. That keeps evaluation from
#: consuming a third of the compute budget while leaving every reported rate at n >= 25.
N_EVAL_CONDITIONS = int(os.environ.get("N_EVAL_CONDITIONS", "50"))
N_EVAL_PERIODIC = int(os.environ.get("N_EVAL_PERIODIC", "25"))
RESET_SUITE_SEED = int(os.environ.get("RESET_SUITE_SEED", "20260817"))
#: Run a paired STOCHASTIC evaluation (same conditions, current exploration noise)
#: alongside each deterministic one. Costs one extra sweep per checkpoint.
EVAL_STOCHASTIC = os.environ.get("EVAL_STOCHASTIC", "1") == "1"
UTD = float(os.environ.get("UTD", "0.5"))             # gradient updates per env decision
#: CAPS-style temporal smoothness weight on the actor's predicted chunk (L_T of
#: Mysore et al., ICRA 2021). 0 disables (the frozen experiment). Motivated by the
#: measured RL-induced finger oscillation: demos move 0.0009/step, the RL policy
#: 0.04/step -- see docs/contracts/g1_piston_rl_induced_oscillation.json. The penalty
#: evaluates ~1e-6 on demo-like motion and ~1e-3 on the pathology, so weights around
#: 0.1-1.0 punish the pump without constraining demonstrated behaviour.
SMOOTH_LAMBDA = float(os.environ.get("SMOOTH_LAMBDA", "0"))
#: Train against the FUNCTIONAL pipette reward (v2): the plunger -- the object's
#: prismatic PistonJoint -- must actually be depressed, and the v1 lift/throw exploit
#: is closed. v1 (transport-only) stays the default so prior runs reproduce exactly.
#: See docs/contracts/g1_piston_plunger_dof.json.
REWARD_V2 = os.environ.get("REWARD_V2", "0") == "1"
#: Reward v3 (review fixes: 3-D grasp with opposition, exploit-free success, object-jerk
#: penalty). Supersedes v2; see docs/contracts/g1_piston_reward_v3_review_fixes.json.
REWARD_V3 = os.environ.get("REWARD_V3", "0") == "1"
#: Feed the critic the action chunk in a truncated temporal (DCT) basis instead of 900
#: raw scalars: measured ~8 -> ~41 transitions per critic input dimension, retaining
#: 99.9994% of trajectory energy on real demonstrations. Critic-side only; the executed
#: action is unchanged. See g1_piston_action_basis.py.
ACTION_BASIS = os.environ.get("ACTION_BASIS", "0") == "1"
#: Give the critic privileged proprioception + object state + phase one-hot alongside
#: the VLM feature (asymmetric actor-critic; the actor still sees only RGB).
CRITIC_STATE = os.environ.get("CRITIC_STATE", "0") == "1"
#: Draw exploration noise with TEMPORAL CORRELATION instead of 900 independent scalars.
#: Measured on the demonstrations: i.i.d. noise at std 0.20 perturbs consecutive steps by
#: 0.2257 rad against the data's own 0.0011 (209x) and puts 80% of its energy in temporal
#: components the task never uses. Correlated noise at the SAME per-step scale cuts the
#: jitter 10.2x and places ~100% of the energy in the subspace the demonstrations occupy.
#: See g1_piston_correlated_policy.py.
CORRELATED_NOISE = os.environ.get("CORRELATED_NOISE", "0") == "1"
#: Encode every demonstration transition once before training instead of paying VLM
#: cache misses inside the update loop. Measured cost ~105 s for 489 transitions; at
#: UTD > 1 the misses otherwise dominate the loop (see the run-2 throughput analysis).
PREWARM_DEMO_CACHE = os.environ.get("PREWARM_DEMO_CACHE", "1") == "1"
#: The step-0 evaluations cost 2 x N_EVAL_PERIODIC x EP_CHUNKS chunk rollouts before a
#: single gradient step. When the warm-start checkpoint has already been scored under the
#: same predicate, that is ~1150 rollouts of pure duplication.
RUN_INIT_EVAL = os.environ.get("RUN_INIT_EVAL", "1") == "1"
#: Optional RL checkpoint to warm-start from (continuation runs). Loaded after the
#: networks are built; this dict is mutated in place so the emitted config sees it.
WARM_CKPT = os.environ.get("WARM_CKPT", "")
WARM_META = {"warm_ckpt": WARM_CKPT} if WARM_CKPT else {}
BATCH = int(os.environ.get("BATCH", "8"))
DEMO_FRAC = float(os.environ.get("DEMO_FRAC", "0.5")) # RLPD offline mix
SEED = int(os.environ.get("SEED", "0"))
EP_CHUNKS = int(os.environ.get("EP_CHUNKS", "23"))    # 23 chunks x 30 x 2 = 1380 env steps
#: RLPD's offline half. The buffer's rewards are baked in at BUILD time, so a run
#: using REWARD_V2 must point at a buffer rebuilt under v2 (see build_demo_buffer.py);
#: otherwise half of every batch carries v1 targets and cancels the v2 press signal.
DEMO_DIR = os.environ.get("DEMO_DIR", "/home/jren313/research/starvla_rl/demo_buffer")

os.makedirs(RUN_DIR, exist_ok=True)
res = {
    "algo": ALGO, "seed": SEED, "config": {
        "max_env_steps": MAX_ENV_STEPS, "eval_every": EVAL_EVERY,
        "n_eval_conditions": N_EVAL_CONDITIONS,
        "n_eval_periodic": N_EVAL_PERIODIC, "reset_suite_seed": RESET_SUITE_SEED,
        "eval_stochastic": EVAL_STOCHASTIC,
        "smooth_lambda": SMOOTH_LAMBDA, "warm_start": WARM_META,
        "reward_version": ("v3_review_fixed" if REWARD_V3
                           else "v2_functional" if REWARD_V2 else "v1_transport"),
        "action_basis": bool(ACTION_BASIS), "critic_state": bool(CRITIC_STATE),
        "correlated_noise": bool(CORRELATED_NOISE),
        "prewarm_demo_cache": bool(PREWARM_DEMO_CACHE),
        "run_init_eval": bool(RUN_INIT_EVAL),
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
    RW2 = _load("g1r2", RL + "g1_piston_reward_v2.py") if REWARD_V2 else None
    RW3 = _load("g1r3", RL + "g1_piston_reward_v3.py") if REWARD_V3 else None
    ABAS = _load("g1ab", RL + "g1_piston_action_basis.py") if ACTION_BASIS else None
    CST = _load("g1cs", RL + "g1_piston_critic_state.py") if CRITIC_STATE else None
    CNZ = _load("g1cn", RL + "g1_piston_correlated_policy.py") if CORRELATED_NOISE else None
    RLSP = _load("g1s", RL + "g1_piston_rl_space.py")
    AFLT = _load("g1af", RL + "g1_piston_action_filter.py")

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
    reward_fn = (RW3.PistonTaskRewardV3(sc, jn) if REWARD_V3
                 else RW2.PistonTaskRewardV2(sc, jn) if REWARD_V2
                 else RW.PistonTaskReward(sc, jn))
    critic_state = CST.CriticStateBuilder(sc, max_chunks=EP_CHUNKS) if CRITIC_STATE else None
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
    from rlinf.envs.isaaclab.tasks.g1_piston_reset import (
        CANONICAL,
        apply_reset_condition,
        build_reset_suite,
        suite_manifest,
    )
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
    # Initialise exploration AT the entropy target's std (0.20 -> log-prob ~ -8.4), so
    # training starts at the alpha equilibrium instead of driving toward it from one
    # side. RLSP.TARGET_ENTROPY_STD and default_target_entropy() are two views of the
    # same operating point.
    INIT_LOGSTD = float(os.environ.get("INIT_LOGSTD", math.log(RLSP.TARGET_ENTROPY_STD)))
    actor_logstd = torch.nn.Parameter(torch.full((30,), INIT_LOGSTD, device=DEV))

    # Critic action-input width: 900 raw, or 180 in the truncated temporal basis.
    A_IN = ABAS.critic_input_dim() if ACTION_BASIS else 30 * H
    # Privileged state is concatenated onto the VLM feature (asymmetric actor-critic).
    C_IN = HID + (CST.STATE_DIM if CRITIC_STATE else 0)
    critic = MultiQHead(C_IN, A_IN, [256, 256], num_q_heads=2).to(DEV)
    target = MultiQHead(C_IN, A_IN, [256, 256], num_q_heads=2).to(DEV)

    # Correlated exploration sampler. Built once; carries the fixed DCT basis.
    corr_noise = (CNZ.CorrelatedChunkNoise(horizon=H, dims=30, device=DEV)
                  if CORRELATED_NOISE else None)

    def critic_action(a):
        """Map a raw action chunk [B,H,30] to the critic's action input [B,A_IN]."""
        return (ABAS.flatten(ABAS.project(a)) if ACTION_BASIS
                else a.reshape(a.shape[0], -1))
    target.load_state_dict(critic.state_dict())
    for p in target.parameters(): p.requires_grad_(False)

    # Why both SAC pilots failed, corrected after measuring the action distribution.
    # See docs/contracts/g1_piston_sac_pilot_v1_collapse.json.
    #
    # ROOT CAUSE: the entropy target was UNREACHABLE. ``-dim(A) = -20`` assumes an
    # unbounded Gaussian, but this action space is tanh-squashed and rescaled to
    # +/-2.2, so its log-density is bounded BELOW at -13.68 (measured; the minimum is
    # interior, at std ~ 0.40, because the tanh Jacobian dominates at both extremes).
    # With ``alpha_loss = -alpha * (logp + target)``, a target below the floor makes the
    # gradient positive across the whole operating range, so alpha falls monotonically,
    # the entropy regulariser dies, and the actor drifts unregularised. That is exactly
    # what both pilots showed: alpha 0.049 -> 0.039 while behaviour oscillated between
    # reach/grasp 1.0 and 0, with actor loss improving throughout.
    #
    # ``default_target_entropy()`` now returns -8.4, inside the achievable range, and
    # exploration is initialised at the matching std (0.20) so training starts at the
    # alpha equilibrium rather than driving toward it from one side.
    ALPHA_INIT = float(os.environ.get("ALPHA_INIT", "0.05"))
    ALPHA_LR = float(os.environ.get("ALPHA_LR", "1e-3"))
    ACTOR_LR = float(os.environ.get("ACTOR_LR", "3e-6"))
    ent = EntropyTemperature(initial_alpha=ALPHA_INIT, alpha_type="softplus",
                             device=DEV).to(DEV)

    # Warm start from an existing RL checkpoint (continuation runs, e.g. adding the
    # smoothness penalty to an already-trained policy). Loads all learned state --
    # action head, critic, target, exploration log-std, alpha -- so optimisation
    # resumes from the checkpoint's operating point rather than from SFT. Env-step
    # counting restarts at 0 for the new run; the origin is recorded in the config.
    if WARM_CKPT:
        wc = torch.load(WARM_CKPT, map_location="cpu", weights_only=False)
        # The POLICY always transfers: it is what carries the learned behaviour, and its
        # shape does not depend on how the critic is conditioned.
        model.action_model.load_state_dict(wc["action_model"])
        with torch.no_grad():
            actor_logstd.copy_(wc["actor_logstd"].to(DEV))
        ent.load_state_dict(wc["alpha"])
        # The CRITIC only transfers when its input space is unchanged. ACTION_BASIS and
        # CRITIC_STATE alter that width (2048+900 = 2948 raw, versus 2048+68+180 = 2296
        # with both on), so a checkpoint predating them cannot be loaded -- and padding
        # or truncating the first layer would be worse than starting fresh, because the
        # surviving weights would be indexed against a different feature layout.
        # A fresh critic is also the correct choice on the merits: it is refitted from
        # the replay buffer within a few hundred updates, whereas a mis-indexed one
        # would emit confident nonsense that the actor would then maximise.
        want = critic.state_dict()["qs.0.net.0.weight"].shape
        got = wc["critic"]["qs.0.net.0.weight"].shape
        if want == got:
            critic.load_state_dict(wc["critic"])
            target.load_state_dict(wc["target"])
            WARM_META["critic_transferred"] = True
        else:
            target.load_state_dict(critic.state_dict())
            WARM_META["critic_transferred"] = False
            WARM_META["critic_reinit_reason"] = (
                f"critic input width changed {tuple(got)} -> {tuple(want)} "
                "(ACTION_BASIS / CRITIC_STATE); policy and alpha still transferred")
        WARM_META["warm_env_steps"] = int(wc.get("env_steps", -1))
        WARM_META["warm_grad_updates"] = int(wc.get("grad_updates", -1))
    # ENTROPY REDUCTION CONVENTION (see tests/unit_tests/
    # test_g1_piston_sac_decision_variable.py, which pins this):
    #
    #   logp_step[h] = sum over the 20 ACTIVE dims          -> per control action
    #   logp_chunk   = mean over the 30 horizon steps       -> per control action
    #   target       = -20                                  -> per control action
    #
    # All three describe the same object. The earlier pilots summed over all
    # 30 x 20 = 600 stochastic scalars while keeping the per-step -20 target, so the
    # entropy term reached ~300 nats against Q ~ 3-7 and dominated the actor objective
    # purely because the policy emits 30 steps at once. Averaging over the horizon keeps
    # regularisation on a per-control-action scale that does not grow with the horizon.
    TARGET_ENTROPY = RLSP.default_target_entropy()  # -8.4, per control action

    # The actor LR is deliberately small: it fine-tunes a converged SFT head, and the
    # collapsed pilot showed the OFT head can be driven off-distribution quickly
    # (actor grad-norm reached 110 at lr 1e-5).
    opt_actor = torch.optim.Adam(
        list(model.action_model.parameters()) + [actor_logstd], lr=ACTOR_LR)
    opt_critic = torch.optim.Adam(critic.parameters(), lr=3e-4)
    opt_alpha = torch.optim.Adam(ent.parameters(), lr=ALPHA_LR)

    # DISCOUNT (review point 3). One RL transition is a whole H=30-step chunk, so the
    # per-step convention gamma=0.99 compounds to 0.99^30 = 0.7397 PER CHUNK. Over a
    # 23-chunk episode that discounts the terminal reward to 0.7397^23 = 0.001 -- one
    # tenth of one percent -- so from the first chunk the success bonus is numerically
    # invisible and the agent is trained to be almost purely myopic. On a long-horizon
    # task whose payoff is entirely terminal (dispense, then place), that alone would
    # prevent the behaviour from ever being learned.
    #
    # GAMMA_CHUNK is therefore specified directly at the chunk level, where the decision
    # actually happens, instead of being derived from a per-step number. The default
    # 0.98 keeps 0.98^23 = 0.63 of the terminal reward visible at episode start
    # (equivalent per-step gamma 0.99933). Set GAMMA_CHUNK to override.
    GAMMA = float(os.environ.get("GAMMA_CHUNK", "0.98"))
    if not 0.0 < GAMMA < 1.0:
        raise SystemExit(f"GAMMA_CHUNK must be in (0,1), got {GAMMA}")
    #: Terminal-reward credit surviving to the first chunk, recorded for auditability.
    TERMINAL_CREDIT = GAMMA ** EP_CHUNKS
    TAU = 0.005

    # Disjoint TRAIN / EVAL initial conditions. The task's own reset is deterministic, so
    # all variation comes from here; the suite is reproducible from RESET_SUITE_SEED and
    # every checkpoint of every algorithm sees exactly the same EVAL conditions.
    TRAIN_CONDITIONS, EVAL_CONDITIONS = build_reset_suite(
        n_train=400, n_eval=N_EVAL_CONDITIONS, seed=RESET_SUITE_SEED)
    res["reset_suite"] = suite_manifest(TRAIN_CONDITIONS, EVAL_CONDITIONS)
    res["setup"] = {
        "hidden": HID, "action_horizon": H, "n_active_dims": N_ACTIVE,
        "target_entropy": TARGET_ENTROPY, "gamma_chunk": round(GAMMA, 5),
        "terminal_credit_at_episode_start": round(TERMINAL_CREDIT, 5),
        "equivalent_per_step_gamma": round(GAMMA ** (1.0 / H), 6),
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
        elif corr_noise is not None:
            # Correlated exploration: sample low-dimensional coefficients and expand
            # them into a temporally smooth chunk, then squash. The pre-squash sample
            # is Gaussian with a structured covariance, so this remains an exact
            # reparameterised draw; the entropy term uses the whitened coefficients'
            # density, which is what the temperature should regulate.
            std_d = torch.exp(logstd)                       # [30]
            zc = torch.randn(b, corr_noise.n_basis, d, device=flat.device)
            eps = corr_noise.expand(zc, std_d)              # [b, H, 30]
            pre = flat.reshape(b, c, d) + eps
            scale = (ACTION_HIGH - ACTION_LOW) / 2.0
            shift = (ACTION_HIGH + ACTION_LOW) / 2.0
            a = (torch.tanh(pre) * scale + shift).reshape(b * c, d)
            # Per-control-action log-prob on the same scale as the i.i.d. path: the
            # coefficient density spread over the horizon, plus the tanh Jacobian on the
            # active dims (the term that actually depends on the sample).
            jac = (torch.log(1 - torch.tanh(pre).pow(2) + 1e-7) * ACT_MASK).sum(dim=-1)
            lp = (CNZ.CorrelatedChunkNoise.logprob_z(zc).unsqueeze(-1) / c) - jac
            lp = lp.reshape(b * c)
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
    def _rollout(cond, save_frames=False, deterministic=True):
        """One deterministic episode from a given initial condition."""
        env.reset(seed=0)                 # the task reset is deterministic ...
        apply_reset_condition(env, cond)  # ... so the condition supplies the variation
        reward_fn.reset()
        bar0 = sc["object"].data.body_pos_w[0, 1].cpu().numpy().copy()
        ret = 0.0; maxlift = 0.0; stages = {}; frames = []
        maxpress = 0.0; maxpress_aligned = 0.0   # plunger, v2 only (0.0 under v1)
        for _ in range(EP_CHUNKS):
            img = get_img()
            if save_frames: frames.append(img)
            _, mean, _ = vlm_feature_and_mean(img)
            a, _ = sample_action(mean, actor_logstd, deterministic=deterministic)
            r, done, info = run_chunk(a[0])
            ret += r
            for k, v in info.get("stages", {}).items():
                stages[k] = stages.get(k, False) or v
            bar = sc["object"].data.body_pos_w[0, 1].cpu().numpy()
            maxlift = max(maxlift, float(bar[2] - bar0[2]))
            maxpress = max(maxpress, float(info.get("max_press_m", 0.0)))
            maxpress_aligned = max(maxpress_aligned,
                                   float(info.get("max_press_aligned_m", 0.0)))
            if done: break
        bar = sc["object"].data.body_pos_w[0, 1].cpu().numpy()
        return {
            "condition": cond.index, "hash": cond.hash(),
            "return": round(float(ret), 3),
            "disp_m": round(float(np.linalg.norm(bar - bar0)), 4),
            # Carry/throw test HORIZONTAL transport; disp_m is a 3-D
            # norm and a vertical fling could clear the threshold on
            # height alone. Record XY explicitly.
            "disp_xy_m": round(float(
                np.linalg.norm((bar - bar0)[:2])), 4),
            "final_dz_m": round(float(bar[2] - bar0[2]), 4),
            "max_lift_m": round(maxlift, 4),
            # Plunger depression, the functional act (v2 reward only; 0.0 under v1).
            "max_press_m": round(maxpress, 5),
            "max_press_aligned_m": round(maxpress_aligned, 5),
            "stages": {k: bool(v) for k, v in stages.items()},
        }, frames

    def evaluate(tag, env_steps, save_video=False, full=False, deterministic=True):
        """Policy over the held-out initial-condition suite.

        ``full`` uses all 50 conditions (final comparison); otherwise the first
        ``N_EVAL_PERIODIC`` of the same fixed suite.

        ``deterministic=False`` re-runs the SAME conditions with the current learned
        exploration noise, to measure the stochastic-vs-deterministic execution gap on
        a contact-rich task (the deterministic policy can lift long before the behaviour
        policy reliably executes the same contact sequence).
        """
        conds = EVAL_CONDITIONS if full else EVAL_CONDITIONS[:N_EVAL_PERIODIC]
        rows = []
        for i, cond in enumerate(conds):
            row, frames = _rollout(cond, save_frames=(save_video and i == 0),
                                   deterministic=deterministic)
            rows.append(row)
            # An eval sweep takes 25-50 min; publish progress so a long run is
            # observable rather than silent until the whole sweep finishes.
            res["eval_progress"] = {"tag": tag, "env_steps": env_steps,
                                    "done": len(rows), "of": len(conds)}
            emit()
            if frames:
                try:
                    import imageio.v2 as imageio
                    imageio.mimsave(f"{RUN_DIR}/{ALGO}_step{env_steps}_cond{cond.index}.mp4",
                                    frames, fps=8)
                except Exception:
                    pass
        n = len(rows)

        def rate(k):
            return round(sum(1 for r in rows if r["stages"].get(k)) / n, 3)

        def wilson(k):
            """95% Wilson interval for a stage rate over n evaluation episodes."""
            import math
            c = sum(1 for r in rows if r["stages"].get(k))
            if n == 0:
                return [0.0, 0.0]
            z = 1.96
            p = c / n
            d = 1 + z * z / n
            centre = (p + z * z / (2 * n)) / d
            half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
            return [round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3)]
        ev = {
            "tag": tag, "env_steps": env_steps,
            "mode": "deterministic" if deterministic else "stochastic",
            "wall_clock_s": round(time.time() - t_start, 1),
            "n_eval_episodes": n,
            # Exploration state at evaluation time, so the deterministic/stochastic gap
            # can be read against the noise that produced it.
            "alpha": float(ent.compute_alpha().item()),
            "action_std_mean": float(torch.exp(actor_logstd).mean().item()),
            "action_std_active_mean": float(
                torch.exp(actor_logstd)[ACT_MASK].mean().item()),
            "full_success_rate": rate("success"), "reach_rate": rate("reach"),
            "grasp_rate": rate("grasp"), "lift_rate": rate("lift"),
            "plate_rate": rate("plate"), "tube_rate": rate("tube"),
            "ci95": {k: wilson(k) for k in
                     ("success", "reach", "grasp", "lift", "plate", "tube")},
            "mean_return": round(float(np.mean([r["return"] for r in rows])), 3),
            "mean_disp_m": round(float(np.mean([r["disp_m"] for r in rows])), 4),
            "mean_max_lift_m": round(float(np.mean([r["max_lift_m"] for r in rows])), 4),
            "per_condition": rows,
        }
        res["evals"].append(ev); emit()
        return ev

    def evaluate_canonical(tag, env_steps):
        """The original single fixed reset, kept as a DIAGNOSTIC.

        One binary observation -- deliberately stored apart from the suite so it can
        never be reported as a success rate.
        """
        row, _ = _rollout(CANONICAL)
        res.setdefault("canonical_diagnostic", []).append(
            {"tag": tag, "env_steps": env_steps, **row})
        emit()
        return row

    # ---------------- SFT baseline: evaluate once, no training ----------------
    if ALGO == "sft_eval":
        evaluate("sft_baseline", 0, save_video=True, full=True)
        evaluate_canonical("sft_baseline", 0)
        res["wall_clock_s"] = round(time.time() - t_start, 1)
        emit("OK"); os._exit(0)

    # ---------------- replay buffers ----------------
    online = deque(maxlen=20000)     # (feat, mean_action, action, reward, next_feat, done)
    demo = []
    #: Privileged critic state per demo transition, when the buffer provides it.
    demo_state, demo_next_state = {}, {}
    if ALGO == "rlpd":
        # Fail loudly rather than train a critic on rewards from the wrong reward
        # version: it would look like a normal run and quietly produce a v1 policy.
        # The buffer's rewards are baked in at BUILD time, so the version it was built
        # with must equal the version being trained -- for EVERY version, not just v2.
        # A mismatch trains the critic on targets from a different objective while the
        # run looks entirely normal, so this fails closed.
        _marker = os.path.join(DEMO_DIR, "_build_status.json")
        _built = "v1"          # buffers predating the marker are v1 by construction
        if os.path.exists(_marker):
            try:
                _built = json.load(open(_marker)).get("reward_version", "v1")
            except Exception:
                _built = "v1"  # a corrupt marker must not read as a newer version
        _want = ("v3_review_fixed" if REWARD_V3
                 else "v2_functional" if REWARD_V2 else "v1")
        # build_demo_buffer writes "v1"; older markers may say "v1_transport".
        _norm = {"v1_transport": "v1"}
        if _norm.get(_built, _built) != _want:
            raise SystemExit(
                f"Demo buffer reward version mismatch: training wants '{_want}' but the "
                f"buffer at {DEMO_DIR} was built with '{_built}'. Rebuild it with "
                f"tools/g1_piston/build_demo_buffer.py using the matching flag "
                "(REWARD_V3=1 / REWARD_V2=1 / neither) and point DEMO_DIR at the result.")
        for fn in sorted(os.listdir(DEMO_DIR)):
            if not fn.endswith(".npz"): continue
            z = np.load(os.path.join(DEMO_DIR, fn))
            imgs, acts, rews = z["images"], z["actions"], z["rewards"]
            cstates = z["critic_state"] if "critic_state" in z.files else None
            if CRITIC_STATE and cstates is None:
                raise SystemExit(
                    f"CRITIC_STATE=1 but {fn} carries no critic_state array. Rebuild "
                    "the buffer with tools/g1_piston/build_demo_buffer.py.")
            for i in range(len(acts) - 1):
                if cstates is not None:
                    demo_state[len(demo)] = cstates[i]
                    demo_next_state[len(demo)] = cstates[i + 1]
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
                # The shipped demo buffer stores images, actions and rewards but NOT
                # simulator state, so the privileged critic vector cannot be
                # reconstructed for demonstration transitions. Feeding zeros would be
                # worse than useless -- the critic would learn that "all-zero state"
                # means "demonstration-quality return", a shortcut it could exploit on
                # online data too. Instead the demo buffer must be rebuilt WITH state
                # (build_demo_buffer.py records it), and until then CRITIC_STATE and
                # RLPD are mutually exclusive rather than silently mismatched.
                dcs, dncs = None, None
                if critic_state is not None:
                    dcs = demo_state.get(idx)
                    dncs = demo_next_state.get(idx)
                    if dcs is None or dncs is None:
                        raise SystemExit(
                            "CRITIC_STATE=1 needs a demo buffer built with simulator "
                            "state. Rebuild with tools/g1_piston/build_demo_buffer.py "
                            "(it records critic_state) and point DEMO_DIR at it.")
                demo_feat_cache[idx] = store(
                    f, aq, torch.tensor(act, dtype=torch.float32), r, nf, naq, d,
                    cs=dcs, ncs=dncs)
            out.append(demo_feat_cache[idx])
        return out

    def collate(items):
        """items: (pooled, action_queries, action, reward, next_pooled, next_aq, done)

        Buffer entries are stored in float16 (see ``store``) and restored to float32
        here, which is where every consumer expects them.
        """
        def cat(i):
            return torch.stack([x[i] for x in items]).to(DEV, dtype=torch.float32)
        feats, aqs, acts = cat(0), cat(1), cat(2)
        rews = torch.tensor([x[3] for x in items], dtype=torch.float32, device=DEV).unsqueeze(-1)
        nfeats, naqs = cat(4), cat(5)
        dones = torch.tensor([x[6] for x in items], dtype=torch.bool, device=DEV).unsqueeze(-1)
        return feats, aqs, acts, rews, nfeats, naqs, dones

    def _fuse(feat, cs):
        """Concatenate the privileged critic state onto the VLM feature.

        Done at STORE time so the replay format, collate, and every consumer stay
        unchanged, and so a stored transition always carries the state that produced
        it -- there is no way for a later change to silently re-pair them.
        """
        if cs is None:
            return feat
        v = torch.as_tensor(cs, dtype=feat.dtype, device=feat.device).view(1, -1)
        return torch.cat([feat, v], dim=-1)

    def store(feat, aq, action, reward, nfeat, naq, done, cs=None, ncs=None):
        """Build one replay entry, holding the big tensors in float16 on CPU.

        The action queries dominate the footprint (H x HID = 30 x 2048 per state, twice
        per transition): at float32 a 500-episode run would hold ~5.5 GB of CPU RAM
        alongside a resident Isaac Sim. They are a deterministic function of the FROZEN
        VLM, so half precision costs nothing that training can observe.
        """
        h = torch.float16
        feat, nfeat = _fuse(feat, cs), _fuse(nfeat, ncs)
        return (feat.squeeze(0).to("cpu", dtype=h), aq.squeeze(0).to("cpu", dtype=h),
                action.detach().to("cpu", dtype=h), float(reward),
                nfeat.squeeze(0).to("cpu", dtype=h), naq.squeeze(0).to("cpu", dtype=h),
                bool(done))

    # ---------------- gradient update ----------------
    def sac_update(batch_items):
        feats, aqs, acts, rews, nfeats, naqs, dones = collate(batch_items)
        B = feats.shape[0]

        # --- critic ---
        with torch.no_grad():
            # next action from the CURRENT policy at the sampled next states
            nmean = head_mean(naqs).float()
            na, nlp = sample_action(nmean, actor_logstd)
            qn = target(nfeats, critic_action(na))
            qmin = qn.min(dim=1, keepdim=True)[0]
            alpha = ent.compute_alpha().detach()
            # mean over the horizon: per-control-action entropy scale (see TARGET_ENTROPY)
            qmin = qmin - alpha * nlp.mean(dim=-1, keepdim=True)
            tq = rews + (~dones) * GAMMA * qmin
        q = critic(feats, critic_action(acts))
        closs = F.mse_loss(q, tq.expand_as(q))
        opt_critic.zero_grad(); closs.backward()
        cgn = torch.nn.utils.clip_grad_norm_(critic.parameters(), 10.0)
        opt_critic.step()

        # --- actor: gradient flows head_mean -> rsample -> critic ---
        mean_b = head_mean(aqs).float()                 # differentiable in OFT head
        a, lp = sample_action(mean_b, actor_logstd)
        qpi = critic(feats, critic_action(a)).min(dim=1, keepdim=True)[0]
        alpha = ent.compute_alpha().detach()
        logp_chunk = lp.mean(dim=-1, keepdim=True)      # per control action
        entropy_term = alpha * logp_chunk
        aloss = (entropy_term - qpi).mean()
        if SMOOTH_LAMBDA > 0:
            # Penalise the DETERMINISTIC squashed chunk, not the sample: the target is
            # the policy's predicted trajectory, and sampling noise would swamp it.
            det = torch.tanh(mean_b) * ((ACTION_HIGH - ACTION_LOW) / 2.0) \
                  + (ACTION_HIGH + ACTION_LOW) / 2.0
            smooth = AFLT.temporal_smoothness_penalty(det, ACT_MASK.cpu()
                                                      if det.device.type == "cpu"
                                                      else ACT_MASK)
            aloss = aloss + SMOOTH_LAMBDA * smooth
        opt_actor.zero_grad(); aloss.backward()
        agn = torch.nn.utils.clip_grad_norm_(
            list(model.action_model.parameters()) + [actor_logstd], 10.0)
        opt_actor.step()

        # --- alpha ---
        alpha_v = ent.compute_alpha()
        alloss = -alpha_v * (logp_chunk.mean().detach() + TARGET_ENTROPY)
        opt_alpha.zero_grad(); alloss.backward(); opt_alpha.step()

        # --- target soft update ---
        with torch.no_grad():
            for tp, op in zip(target.parameters(), critic.parameters()):
                tp.data.mul_(1 - TAU).add_(op.data, alpha=TAU)

        # Relative magnitudes are logged explicitly: the entropy term must not dwarf the
        # learned Q signal simply because the policy emits 30 steps at once.
        q_term = float(qpi.mean().item())
        ent_term = float(entropy_term.mean().item())
        return {
            "critic_loss": float(closs.item()), "actor_loss": float(aloss.item()),
            "alpha_loss": float(alloss.item()), "alpha": float(alpha_v.item()),
            "q_mean": float(q.mean().item()), "target_q_mean": float(tq.mean().item()),
            "logprob_per_step": float(logp_chunk.mean().item()),
            "actor_q_term": q_term,
            "actor_entropy_term": ent_term,
            "abs_alpha_logprob": abs(ent_term),
            "entropy_to_q_ratio": round(abs(ent_term) / max(abs(q_term), 1e-6), 4),
            "critic_gn": float(cgn), "actor_gn": float(agn),
        }

    # ---------------- training loop ----------------
    env_steps = 0; grad_updates = 0; episode = 0
    n_online_samples = 0; n_demo_samples = 0
    # NOTE (deliberate, do not "fix" mid-experiment): ``next_eval`` starts at 0, so the
    # first training episode trips the periodic branch immediately (env_steps 1380 >= 0)
    # and runs one extra 25-condition sweep before settling onto the intended schedule
    # (138000, 276000, ...). That costs ~24 min per arm and yields a harmless extra
    # early data point.
    #
    # It is left as-is for the SAC/RLPD comparison because the running SAC arm loaded
    # this file at startup: changing it now would give RLPD a different evaluation
    # cadence than SAC, and matched conditions matter more than 24 minutes. Set this to
    # EVAL_EVERY once both arms of the current comparison have finished.
    next_eval = 0

    # Pre-warm the demonstration VLM cache BEFORE training. Measured: one vlm_encode
    # costs 107.5 ms, a demo cache MISS costs two of them, and RLPD draws BATCH*DEMO_FRAC
    # demo transitions on every gradient update. At UTD 8 that is up to 8 misses per
    # environment decision while the cache fills -- which is why raising UTD slowed
    # collection 16x instead of being nearly free (run 2). Paying the whole cache up
    # front is a bounded one-time cost (489 transitions x 2 encodes = ~105 s) after
    # which demo_batch really is a dict lookup.
    if demo and PREWARM_DEMO_CACHE:
        _t0 = time.time()
        for _idx in range(len(demo)):
            if _idx not in demo_feat_cache:
                _img, _act, _r, _nimg, _d = demo[_idx]
                _aq, _f = vlm_encode(_img)
                _naq, _nf = vlm_encode(_nimg)
                _dcs = demo_state.get(_idx) if critic_state is not None else None
                _dncs = demo_next_state.get(_idx) if critic_state is not None else None
                demo_feat_cache[_idx] = store(
                    _f, _aq, torch.tensor(_act, dtype=torch.float32), _r, _nf, _naq, _d,
                    cs=_dcs, ncs=_dncs)
        res["demo_cache_prewarm_s"] = round(time.time() - _t0, 1)
        res["demo_cache_entries"] = len(demo_feat_cache)
        emit()

    if RUN_INIT_EVAL:
        evaluate("init", 0, save_video=True)   # step-0 behavior, same suite
        if EVAL_STOCHASTIC:
            evaluate("init_stochastic", 0, deterministic=False)

    while env_steps < MAX_ENV_STEPS:
        # Training initial conditions come from the TRAIN split only; the EVAL split is
        # never seen during training.
        train_cond = TRAIN_CONDITIONS[episode % len(TRAIN_CONDITIONS)]
        env.reset(seed=0)
        apply_reset_condition(env, train_cond)
        reward_fn.reset()
        bar0 = sc["object"].data.body_pos_w[0, 1].cpu().numpy().copy()
        ep_ret = 0.0; ep_stages = {}; maxlift = 0.0
        img = get_img()
        feat, mean, aq = vlm_feature_and_mean(img)
        if critic_state is not None:
            critic_state.reset()
        cs = (critic_state.build(stages=ep_stages, chunk=0)
              if critic_state is not None else None)

        for c in range(EP_CHUNKS):
            a, _ = sample_action(mean, actor_logstd)          # stochastic: explore
            r, done, info = run_chunk(a[0])
            env_steps += H * 2
            nimg = get_img()
            nfeat, nmean, naq = vlm_feature_and_mean(nimg)
            ncs = (critic_state.build(stages=info.get("stages", {}), chunk=c + 1)
                   if critic_state is not None else None)
            online.append(store(feat, aq, a[0], r, nfeat, naq, done, cs=cs, ncs=ncs))
            ep_ret += r
            for k, v in info.get("stages", {}).items():
                ep_stages[k] = ep_stages.get(k, False) or v
            bar = sc["object"].data.body_pos_w[0, 1].cpu().numpy()
            maxlift = max(maxlift, float(bar[2] - bar0[2]))
            feat, mean, aq = nfeat, nmean, naq
            cs = ncs

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
            "episode": episode, "condition": train_cond.index,
            "condition_hash": train_cond.hash(), "env_steps": env_steps,
            "return": round(float(ep_ret), 3),
            "disp_m": round(float(np.linalg.norm(bar - bar0)), 4),
            # Carry/throw test HORIZONTAL transport; disp_m is a 3-D
            # norm and a vertical fling could clear the threshold on
            # height alone. Record XY explicitly.
            "disp_xy_m": round(float(
                np.linalg.norm((bar - bar0)[:2])), 4),
            "final_dz_m": round(float(bar[2] - bar0[2]), 4),
            "max_lift_m": round(maxlift, 4),
            "stages": {k: bool(v) for k, v in ep_stages.items()},
        })
        episode += 1; emit()

        if env_steps >= next_eval:
            evaluate("periodic", env_steps, save_video=True)
            # Paired stochastic pass over the SAME conditions: tests whether exploration
            # noise destroys sustained contact while the mean policy has the skill.
            if EVAL_STOCHASTIC:
                evaluate("periodic_stochastic", env_steps, deterministic=False)
            evaluate_canonical("periodic", env_steps)
            next_eval += EVAL_EVERY
            ck = {
                "action_model": model.action_model.state_dict(),
                "critic": critic.state_dict(), "target": target.state_dict(),
                "actor_logstd": actor_logstd.detach().cpu(),
                "alpha": ent.state_dict(),
                "env_steps": env_steps, "grad_updates": grad_updates,
            }
            torch.save(ck, f"{RUN_DIR}/{ALGO}_ckpt_latest.pt")
            # Keep every scheduled checkpoint, not just the latest: intermediate
            # checkpoints are needed for retrospective analysis and replication.
            torch.save(ck, f"{RUN_DIR}/{ALGO}_ckpt_step{env_steps}.pt")
            res["last_checkpoint"] = f"{RUN_DIR}/{ALGO}_ckpt_latest.pt"
            res.setdefault("checkpoints", []).append(
                {"env_steps": env_steps, "grad_updates": grad_updates,
                 "path": f"{RUN_DIR}/{ALGO}_ckpt_step{env_steps}.pt"})
            # frozen-ness spot check
            with torch.no_grad():
                vlm_now = dict(list(model.qwen_vl_interface.named_parameters())[:5])
                oft_now = dict(list(model.action_model.named_parameters())[:5])
                res["vlm_unchanged"] = all(
                    torch.equal(vlm_ref[n].cpu(), vlm_now[n].detach().cpu()) for n in vlm_ref)
                res["oft_changed"] = any(
                    not torch.equal(oft_ref[n].cpu(), oft_now[n].detach().cpu()) for n in oft_ref)
            emit()

    evaluate("final", env_steps, save_video=True, full=True)
    if EVAL_STOCHASTIC:
        evaluate("final_stochastic", env_steps, full=True, deterministic=False)
    evaluate_canonical("final", env_steps)
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
