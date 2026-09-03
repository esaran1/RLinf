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

"""Behaviour-clone the executable demonstrations onto the OFT action head.

Why this exists
---------------
Every RL run in this project started from an SFT checkpoint that scores **0.00
deterministic success** and ~1% grasp under the corrected predicate, and tried to
improve it with a reward that (until v3) measured xy proximity rather than grasping.
None produced a policy that performs the task.

The demonstrations, however, DO succeed: replayed through the frozen mapper and
retargeter, **16 of 22** executable episodes reach grasp -> lift -> plate. Those
trajectories are the most reliable signal available, and imitating them directly is the
shortest path to a policy that actually manipulates the object.

This is deliberately NOT reinforcement learning. It is supervised regression of the OFT
action head onto demonstration action chunks, conditioned on the same VLM features the
policy sees at deployment:

    loss = MSE( head(vlm_features(image)), demonstrated_chunk )   over active dims

Design choices, and why
-----------------------
* **The VLM stays frozen.** Only ``model.action_model`` trains, exactly as in the RL
  runs, so the resulting checkpoint is drop-in compatible with eval_checkpoint.py and
  render_rollouts.py and can be scored under the same frozen suite.
* **Features are cached once.** A vlm_encode costs ~107 ms; caching all demonstration
  frames up front (measured ~135 s for 489 transitions) makes each epoch nearly free.
  This is the same fix that took RL training from 72 s to 1.23 s per update.
* **Only ACTIVE dims are regressed.** The frozen dims are held at their dataset values
  by the action mask, so including them would dilute the loss with constants.
* **Loss is on the NORMALIZED chunk**, matching the space the policy's head emits.
  The demonstration buffer stores PHYSICAL actions (verified against the recorded
  act_ep*.npy), so targets are passed through the Q99 normaliser first; denormalisation
  then happens downstream in the same code path as every rollout.

Environment:
    OUTF        status/report JSON (required)
    RUN_DIR     checkpoint directory (required)
    DEMO_DIR    demonstration buffer (default: demo_buffer_v3)
    EPOCHS      passes over the demonstration set (default 200)
    BC_LR       learning rate for the action head (default 1e-4)
    BATCH       chunks per gradient step (default 16)
    EPISODES    optional comma-separated subset, e.g. only the lifting episodes
"""
import json
import os
import sys
import time
import traceback

import numpy as np

OUT = os.environ["OUTF"]
RUN_DIR = os.environ["RUN_DIR"]
DEMO_DIR = os.environ.get("DEMO_DIR",
                          "/home/jren313/research/starvla_rl/demo_buffer_v3")
EPOCHS = int(os.environ.get("EPOCHS", "200"))
BC_LR = float(os.environ.get("BC_LR", "1e-4"))
BATCH = int(os.environ.get("BATCH", "16"))
SAVE_EVERY = int(os.environ.get("SAVE_EVERY", "25"))

os.makedirs(RUN_DIR, exist_ok=True)
res = {"_status": "RUNNING", "config": {
    "demo_dir": DEMO_DIR, "epochs": EPOCHS, "lr": BC_LR, "batch": BATCH,
    "method": "behaviour cloning, frozen VLM, OFT head only"}}


def emit(s="RUNNING"):
    res["_status"] = s
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2, default=str)


try:
    os.environ.pop("DISPLAY", None)
    import random

    import torch
    import torch.nn.functional as F

    sys.path.insert(0, "/home/jren313/research/starvla_rl/starVLA")
    from omegaconf import OmegaConf

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    BASE = "/home/jren313/research/starvla_rl/checkpoints/g1-longhorizon-oft-v1"
    CFGY, STATS = f"{BASE}/config.yaml", f"{BASE}/dataset_statistics.json"
    SFT = f"{BASE}/final_model/pytorch_model.pt"
    TASK = ("pick up the piston with the right hand, inject it into the tube held by "
            "the left hand, then move it over the hole plate.")

    import importlib.util as ilu

    def _load(m, p):
        sp = ilu.spec_from_file_location(m, p)
        mo = ilu.module_from_spec(sp)
        sys.modules[m] = mo
        sp.loader.exec_module(mo)
        return mo

    RL = "/home/jren313/research/starvla_rl/RLinf/rlinf/envs/isaaclab/tasks/"
    RLSP = _load("g1s", RL + "g1_piston_rl_space.py")
    Q99 = _load("g1n", RL + "g1_piston_norm.py").Q99ActionNormalizer

    from deployment.model_server.tools.image_tools import to_pil_preserve
    from starVLA.model.framework.base_framework import build_framework
    from starVLA.training.trainer_utils.trainer_tools import resize_images

    nrm = Q99.from_dataset_statistics(STATS)
    mcfg = OmegaConf.load(CFGY)
    model = build_framework(mcfg)
    model.load_state_dict(torch.load(SFT, map_location="cpu", weights_only=False),
                          strict=False)
    model = model.to("cuda")
    DEV = "cuda"
    H = int(model.action_horizon)

    # Freeze everything except the OFT action head -- identical to the RL setup, so the
    # produced checkpoint loads in eval_checkpoint.py and render_rollouts.py unchanged.
    for p_ in model.parameters():
        p_.requires_grad_(False)
    for p_ in model.action_model.parameters():
        p_.requires_grad_(True)
    model.eval()

    ACT_MASK = RLSP.build_active_mask().to(DEV)          # [30] bool
    res["n_active_dims"] = int(ACT_MASK.sum())
    emit()

    def vlm_encode(img_np):
        """Frozen backbone forward. Returns action queries [1,H,HID]."""
        with torch.no_grad():
            imgs = [to_pil_preserve([img_np])]
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
                                             output_hidden_states=True,
                                             return_dict=True)
            hs = qo.hidden_states[-1]
            return hs[:, -model.chunk_len:, :].float()

    def head_mean(aq):
        """Differentiable OFT head forward, IDENTICAL to train_sac.py's head_mean.

        Using any other forward here would train a different function than the one
        eval_checkpoint.py and render_rollouts.py evaluate, so the checkpoint would
        score differently than it was trained -- silently.
        """
        with torch.autocast("cuda", dtype=torch.float32):
            return model.action_model.predict_action(aq)

    # ---------------- load demonstrations and cache their VLM features -------------
    import glob

    want = None
    if os.environ.get("EPISODES"):
        want = {int(x) for x in os.environ["EPISODES"].split(",") if x.strip()}
    pairs = []          # (action_queries [H,HID], target_chunk [H,30])
    eps_used = []
    t0 = time.time()
    for fp in sorted(glob.glob(f"{DEMO_DIR}/*.npz")):
        ep = int(os.path.basename(fp)[2:5])
        if want is not None and ep not in want:
            continue
        z = np.load(fp)
        imgs, acts = z["images"], z["actions"]
        # The final stored observation has no action chunk paired with it.
        for i in range(len(acts) - 1):
            aq = vlm_encode(imgs[i].astype(np.uint8))
            # The buffer stores PHYSICAL actions (verified: they match the recorded
            # act_ep*.npy exactly, and the RL trainer denormalises the head's output
            # before mapping to the simulator). The head therefore emits NORMALIZED
            # actions, so the regression target must be normalised too -- training on
            # raw physical values would fit a policy wrong by the whole transform.
            tgt = nrm.normalize(torch.as_tensor(acts[i], dtype=torch.float32))
            pairs.append((aq.squeeze(0).cpu(), tgt))
        eps_used.append(ep)
    res["episodes_used"] = eps_used
    res["n_transitions"] = len(pairs)
    res["feature_cache_s"] = round(time.time() - t0, 1)
    emit()

    if not pairs:
        raise SystemExit(f"no demonstration transitions found in {DEMO_DIR}")

    opt = torch.optim.Adam(model.action_model.parameters(), lr=BC_LR)
    res["train_log"] = []

    # ---------------- supervised training ------------------------------------------
    t_train = time.time()
    for epoch in range(EPOCHS):
        random.shuffle(pairs)
        tot, nb = 0.0, 0
        for i in range(0, len(pairs), BATCH):
            batch = pairs[i:i + BATCH]
            aq = torch.stack([b[0] for b in batch]).to(DEV)
            tgt = torch.stack([b[1] for b in batch]).to(DEV)
            pred = head_mean(aq)
            # Regress only the ACTIVE dims; the frozen ones are constants supplied
            # downstream and would otherwise dilute the loss.
            loss = F.mse_loss(pred[..., ACT_MASK], tgt[..., ACT_MASK])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.action_model.parameters(), 10.0)
            opt.step()
            tot += float(loss)
            nb += 1
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            res["train_log"].append({"epoch": epoch, "mse": round(tot / max(nb, 1), 6),
                                     "elapsed_s": round(time.time() - t_train, 1)})
            emit()
        if (epoch + 1) % SAVE_EVERY == 0 or epoch == EPOCHS - 1:
            # Saved in the SAME format the RL trainer writes, so every existing
            # evaluation and rendering tool loads it without modification.
            ck = {"action_model": model.action_model.state_dict(),
                  "actor_logstd": torch.full((30,), float(np.log(0.05))),
                  "env_steps": 0, "grad_updates": (epoch + 1) * max(nb, 1),
                  "bc_epoch": epoch + 1}
            torch.save(ck, f"{RUN_DIR}/bc_ckpt_latest.pt")
            torch.save(ck, f"{RUN_DIR}/bc_ckpt_epoch{epoch + 1}.pt")
            res["last_checkpoint"] = f"{RUN_DIR}/bc_ckpt_latest.pt"
            emit()

    res["final_mse"] = res["train_log"][-1]["mse"] if res["train_log"] else None
    res["train_s"] = round(time.time() - t_train, 1)
    emit("OK")
    os._exit(0)
except SystemExit as e:
    res["error"] = str(e)
    emit("ERROR")
    raise
except Exception as e:  # noqa: BLE001
    res["error"] = f"{type(e).__name__}: {e}"
    res["tb"] = traceback.format_exc()
    emit("ERROR")
    os._exit(1)
