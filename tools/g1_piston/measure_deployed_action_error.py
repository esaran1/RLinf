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

"""Measure a checkpoint's action error AT THE POINT OF EXECUTION.

This is the instrument that found both behaviour-cloning defects and that characterises
the RL regression. It walks a demonstration's own frames and runs the FULL deployed
pipeline -- vlm_encode -> head -> tanh squash -> frozen-dim mask -> denormalise -- then
compares against the demonstrated physical action, in degrees.

Why it exists: an earlier probe compared the head's output to the demonstration WITHOUT
the deployment squash. It reported 1.15 degrees for a policy that was 12.4 degrees off
where it counted, and its conclusion ("the policy cannot learn this from one RGB frame")
was wrong. A fidelity measurement is only meaningful at the point of execution.

Reference values on demonstration ep046:

    checkpoint                     before squash   AFTER squash (deployed)
    behaviour cloning, grasp 1.00      8.068              0.403
    RL run 4, grasp 0.00               ~0.4              13.773
    RL run 5, grasp 0.00                -                11.542

A large deployed error with a small raw-weight change is the signature of a policy being
flattened toward the squash's linear centre, not one that learned something different.

Usage:
    OUTF=<json> CKPT=<checkpoint> python measure_deployed_action_error.py
"""
import json, os, sys, glob, traceback
import numpy as np
OUT=os.environ["OUTF"]; CKPT=os.environ["CKPT"]
res={"_status":"RUNNING","ckpt":CKPT}
def emit(s="RUNNING"):
    res["_status"]=s
    with open(OUT,"w") as f: json.dump(res,f,indent=2,default=str)
try:
    os.environ.pop("DISPLAY",None)
    import torch
    sys.path.insert(0,"/home/jren313/research/starvla_rl/starVLA")
    from omegaconf import OmegaConf
    import importlib.util as ilu
    def _load(m,p):
        sp=ilu.spec_from_file_location(m,p); mo=ilu.module_from_spec(sp)
        sys.modules[m]=mo; sp.loader.exec_module(mo); return mo
    RL="/home/jren313/research/starvla_rl/RLinf/rlinf/envs/isaaclab/tasks/"
    RLSP=_load("g1s",RL+"g1_piston_rl_space.py")
    Q99=_load("g1n",RL+"g1_piston_norm.py").Q99ActionNormalizer
    BASE="/home/jren313/research/starvla_rl/checkpoints/g1-longhorizon-oft-v1"
    SFT=f"{BASE}/final_model/pytorch_model.pt"
    TASK=("pick up the piston with the right hand, inject it into the tube held by "
          "the left hand, then move it over the hole plate.")
    from deployment.model_server.tools.image_tools import to_pil_preserve
    from starVLA.model.framework.base_framework import build_framework
    from starVLA.training.trainer_utils.trainer_tools import resize_images
    nrm=Q99.from_dataset_statistics(f"{BASE}/dataset_statistics.json")
    model=build_framework(OmegaConf.load(f"{BASE}/config.yaml"))
    model.load_state_dict(torch.load(SFT,map_location="cpu",weights_only=False),strict=False)
    model=model.to("cuda").eval(); DEV="cuda"
    ck=torch.load(CKPT,map_location="cpu",weights_only=False)
    model.action_model.load_state_dict(ck["action_model"]); model=model.to(DEV).eval()

    ACT_MASK=RLSP.build_active_mask().to(DEV)
    # Identical to eval_checkpoint.py lines 159-161.
    ACTION_LOW, ACTION_HIGH = -2.2, 2.2
    FROZEN_V=RLSP.load_frozen_values(f"{BASE}/dataset_statistics.json").to(DEV).float()
    res["action_bounds"]=[ACTION_LOW, ACTION_HIGH]

    def vlm_encode(img_np):
        with torch.no_grad():
            imgs=[to_pil_preserve([img_np])]
            size=getattr(model.config.datasets.vla_data,"obs_image_size",None)
            if size: imgs=resize_images(imgs,target_size=size)
            toks=model.action_token*model.chunk_len
            instr=TASK+(f" Please predict the next {model.chunk_len} robot actions:"
                        f" <action>{toks}<action>.")
            qi=model.qwen_vl_interface.build_qwenvl_inputs(images=imgs,instructions=[instr])
            with torch.autocast("cuda",dtype=torch.bfloat16):
                qo=model.qwen_vl_interface(**qi,output_attentions=False,
                                           output_hidden_states=True,return_dict=True)
                lh=qo.hidden_states[-1]
            aq=model._gather_action_token_embeddings(lh,qi.get("input_ids",None),
                                                     action_token_id=model.action_token_id)
            with torch.autocast("cuda",dtype=torch.float32):
                return model.action_model.predict_action(aq.detach().float()).float()

    def deployed_physical(mean):
        """EXACTLY eval_checkpoint.sample_action(deterministic) + denormalize."""
        b,c,d=mean.shape
        flat=mean.reshape(b*c,d)
        scale=(ACTION_HIGH-ACTION_LOW)/2.0; shift=(ACTION_HIGH+ACTION_LOW)/2.0
        a=torch.tanh(flat)*scale+shift
        a=torch.where(ACT_MASK,a,FROZEN_V.expand_as(a))
        return nrm.denormalize(a.reshape(b,c,d).detach().cpu())

    z=np.load("/home/jren313/research/starvla_rl/demo_buffer_v3/ep046.npz")
    imgs,acts=z["images"],z["actions"]
    am=ACT_MASK.cpu()
    rows=[]; 
    raw_mean_abs=[]; squashed_err=[]; nosquash_err=[]
    for i in range(min(8,len(acts)-1)):
        mean=vlm_encode(imgs[i].astype(np.uint8))
        phys=deployed_physical(mean)[0]                 # [H,30] physical, deployed path
        tgt=torch.as_tensor(acts[i],dtype=torch.float32)  # [H,30] physical demo
        # what the head emits BEFORE squashing, denormalised directly
        nosq=nrm.denormalize(mean[0].detach().cpu())
        e_sq=(phys-tgt).abs()[:,am]; e_ns=(nosq-tgt).abs()[:,am]
        raw_mean_abs.append(float(mean.abs().mean()))
        squashed_err.append(float(e_sq.mean())); nosquash_err.append(float(e_ns.mean()))
        rows.append({"i":i,"raw_head_abs_mean":round(float(mean.abs().mean()),4),
                     "raw_head_max":round(float(mean.abs().max()),4),
                     "deployed_err_rad":round(float(e_sq.mean()),4),
                     "nosquash_err_rad":round(float(e_ns.mean()),4),
                     "demo_abs_mean":round(float(tgt[:,am].abs().mean()),4),
                     "deployed_abs_mean":round(float(phys[:,am].abs().mean()),4)})
    res["frames"]=rows
    res["summary"]={
        "deployed_path_err_rad":round(float(np.mean(squashed_err)),4),
        "deployed_path_err_deg":round(float(np.mean(squashed_err))*57.2958,3),
        "nosquash_err_rad":round(float(np.mean(nosquash_err)),4),
        "nosquash_err_deg":round(float(np.mean(nosquash_err))*57.2958,3),
        "raw_head_abs_mean":round(float(np.mean(raw_mean_abs)),4),
    }
    emit("OK"); os._exit(0)
except Exception as e:
    res["error"]=f"{type(e).__name__}: {e}"; res["tb"]=traceback.format_exc()
    emit("ERROR"); os._exit(1)
