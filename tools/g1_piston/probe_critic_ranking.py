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
"""Does run 6's critic rate the DRIFTED policy above the BC policy?

No simulator. On demonstration ep046 frames, with run 6's own critic (and its target):
    Q(a_demo)   demonstration action (executed/normalised space)  env return 11.88 (ep mean)
    Q(a_BC)     BC head's deterministic chunk                     env: grasp 1.00, return 11.9
    Q(a_run6)   run 6 head's deterministic chunk (step 60840)     env: grasp 0.04, return ~-0.6
Prediction under critic exploitation: Q(a_run6) > Q(a_BC) although the environment scores
run 6 far lower. Also reports the critic's gradient direction: does moving from a_BC toward
a_run6 raise Q (i.e. is the drift the ascent direction the actor followed)?
"""
import json, os, sys, traceback, numpy as np
OUT=os.environ["OUTF"]; res={"_status":"RUNNING"}
def emit(s="RUNNING"):
    res["_status"]=s
    with open(OUT,"w") as f: json.dump(res,f,indent=2,default=str)
try:
    os.environ.pop("DISPLAY",None)
    import torch
    sys.path.insert(0,"/home/jren313/research/starvla_rl/starVLA"); sys.path.insert(0,"/home/jren313/research/starvla_rl/RLinf")
    from omegaconf import OmegaConf
    import importlib.util as ilu
    def _load(m,p):
        sp=ilu.spec_from_file_location(m,p); mo=ilu.module_from_spec(sp); sys.modules[m]=mo; sp.loader.exec_module(mo); return mo
    RL="/home/jren313/research/starvla_rl/RLinf/rlinf/envs/isaaclab/tasks/"
    RLSP=_load("g1s",RL+"g1_piston_rl_space.py"); ABAS=_load("g1ab",RL+"g1_piston_action_basis.py")
    Q99=_load("g1n",RL+"g1_piston_norm.py").Q99ActionNormalizer
    from rlinf.models.embodiment.modules.q_head import MultiQHead
    BASE="/home/jren313/research/starvla_rl/checkpoints/g1-longhorizon-oft-v1"
    TASK=("pick up the piston with the right hand, inject it into the tube held by the left hand, then move it over the hole plate.")
    from deployment.model_server.tools.image_tools import to_pil_preserve
    from starVLA.model.framework.base_framework import build_framework
    from starVLA.training.trainer_utils.trainer_tools import resize_images
    nrm=Q99.from_dataset_statistics(f"{BASE}/dataset_statistics.json")
    model=build_framework(OmegaConf.load(f"{BASE}/config.yaml"))
    model.load_state_dict(torch.load(f"{BASE}/final_model/pytorch_model.pt",map_location="cpu",weights_only=False),strict=False)
    DEV="cuda"; model=model.to(DEV).eval()
    for p in model.parameters(): p.requires_grad_(False)
    H=int(model.action_horizon); HID=int(model.qwen_vl_interface.model.config.hidden_size)
    am=RLSP.build_active_mask().to(DEV); LOW,HIGH=-2.2,2.2; SC=(HIGH-LOW)/2; SH=(HIGH+LOW)/2
    FROZEN=RLSP.load_frozen_values(f"{BASE}/dataset_statistics.json").to(DEV).float()
    S=os.environ["SCRATCH"]
    BC=torch.load("/home/jren313/research/starvla_rl/checkpoints/g1_piston_bc_working/bc_ckpt_latest.pt",map_location="cpu",weights_only=False)["action_model"]
    R6=torch.load(f"{S}/runs/rl6/rlpd_ckpt_step60840.pt",map_location="cpu",weights_only=False)
    C_IN=HID+68; A_IN=ABAS.critic_input_dim()
    critic=MultiQHead(C_IN,A_IN,[256,256],num_q_heads=2).to(DEV); critic.load_state_dict(R6["critic"]); critic.eval()
    target=MultiQHead(C_IN,A_IN,[256,256],num_q_heads=2).to(DEV); target.load_state_dict(R6["target"]); target.eval()

    def encode(img):
        with torch.no_grad():
            imgs=[to_pil_preserve([img])]
            size=getattr(model.config.datasets.vla_data,"obs_image_size",None)
            if size: imgs=resize_images(imgs,target_size=size)
            toks=model.action_token*model.chunk_len
            instr=TASK+f" Please predict the next {model.chunk_len} robot actions: <action>{toks}<action>."
            qi=model.qwen_vl_interface.build_qwenvl_inputs(images=imgs,instructions=[instr])
            with torch.autocast("cuda",dtype=torch.bfloat16):
                qo=model.qwen_vl_interface(**qi,output_attentions=False,output_hidden_states=True,return_dict=True)
                lh=qo.hidden_states[-1]
            aq=model._gather_action_token_embeddings(lh,qi.get("input_ids"),action_token_id=model.action_token_id)
            m=qi["attention_mask"].to(lh.dtype).unsqueeze(-1)
            pooled=(lh*m).sum(1)/m.sum(1).clamp(min=1e-6)
            return aq.float(), pooled.float()
    def head(aq):
        with torch.no_grad(), torch.autocast("cuda",dtype=torch.float32):
            return model.action_model.predict_action(aq).float()
    def squash(mean):
        a=torch.tanh(mean)*SC+SH; return torch.where(am,a,FROZEN.expand_as(a))
    def q(net,feats,a):
        with torch.no_grad(): return net(feats,ABAS.flatten(ABAS.project(a))).min(dim=1)[0]

    z=np.load("/home/jren313/research/starvla_rl/demo_buffer_v3/ep046.npz")
    rows=[]; feats=[]; aqs=[]
    for i in range(len(z["actions"])-1):
        aq,pooled=encode(z["images"][i].astype(np.uint8))
        cs=torch.as_tensor(z["critic_state"][i],dtype=torch.float32,device=DEV).view(1,-1)
        feats.append(torch.cat([pooled,cs],-1)); aqs.append(aq)
    F_=torch.cat(feats); AQ=torch.cat(aqs)
    a_demo=nrm.normalize(torch.as_tensor(z["actions"][:len(aqs)],dtype=torch.float32)).to(DEV)
    model.action_model.load_state_dict(BC);  a_bc=squash(head(AQ))
    model.action_model.load_state_dict(R6["action_model"]); a_r6=squash(head(AQ))
    out={}
    for name,net in (("critic",critic),("target",target)):
        qd,qb,qr=q(net,F_,a_demo),q(net,F_,a_bc),q(net,F_,a_r6)
        # direction test: interpolate from BC toward run 6
        line=[float(q(net,F_,a_bc+t*(a_r6-a_bc)).mean()) for t in (0.0,0.25,0.5,0.75,1.0)]
        out[name]={"Q_demo":float(qd.mean()),"Q_bc":float(qb.mean()),"Q_run6":float(qr.mean()),
                   "frac_frames_Q_run6_gt_Q_bc":float((qr>qb).float().mean()),
                   "Q_along_bc_to_run6":line}
    res["n_frames"]=int(F_.shape[0]); res["by_net"]=out
    res["deployed_err_deg"]={"bc_vs_demo":float(((nrm.denormalize(a_bc.cpu())-torch.as_tensor(z["actions"][:len(aqs)])).abs()[...,am.cpu()]).mean())*57.3,
                             "run6_vs_demo":float(((nrm.denormalize(a_r6.cpu())-torch.as_tensor(z["actions"][:len(aqs)])).abs()[...,am.cpu()]).mean())*57.3}
    res["environment_truth"]={"bc":"grasp 1.00, lift 0.80, return 11.935 (scorer of record)","run6_step60840":"in-trainer grasp 0.04, lift 0.00 (scorer of record pending)","demo_ep_returns_mean":11.88}
    emit("OK"); os._exit(0)
except Exception as e:
    res["error"]=f"{type(e).__name__}: {e}"; res["tb"]=traceback.format_exc(); emit("ERROR"); os._exit(1)
