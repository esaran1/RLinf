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
"""Is the critic action-sensitive, and does it rank the BC chunk above worse chunks?

No simulator. On demonstration ep046 frames, with the critic (and target) stored in the
checkpoint given by CKPT, score:
    Q(a_BC)       the BC head's deterministic chunk        env: grasp 1.00, return 11.9
    Q(a_policy)   the checkpoint's own deterministic chunk  (base + residual if present)
    Q(a_pert_k)   a_BC perturbed by exploration-scale noise (PERT_STD, K draws)
    Q(a_demo)     the demonstration action

H10 (run8_preregistration.json): Q(a_BC) > Q(a_pert) on >= 70% of (frame, draw) pairs,
and |Q(a_BC) - Q(a_policy)| / |Q(a_BC)| > 0.10. Run 6's critic had Q(drifted) > Q(BC)
on 100% of frames with a 2% gap: action-insensitive and mis-sloped.

The ensemble size is inferred from the checkpoint, the aggregation matches the run's
config (min over all heads unless the run record says otherwise), and a residual policy
in the checkpoint is applied when forming a_policy.
Env: OUTF, CKPT, [RUNJSON] (for aggregation/config), [PERT_STD=0.15], [K=8]
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
    def infer_num_q(sd):
        """Number of Q heads in a MultiQHead state dict (keys 'qs.<i>....')."""
        return 1+max(int(k.split(".")[1]) for k in sd if k.startswith("qs."))
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
    R6=torch.load(os.environ["CKPT"],map_location="cpu",weights_only=False)
    C_IN=HID+68; A_IN=ABAS.critic_input_dim()
    NQ=infer_num_q(R6["critic"]); res["num_q"]=NQ
    critic=MultiQHead(C_IN,A_IN,[256,256],num_q_heads=NQ).to(DEV); critic.load_state_dict(R6["critic"]); critic.eval()
    target=MultiQHead(C_IN,A_IN,[256,256],num_q_heads=NQ).to(DEV); target.load_state_dict(R6["target"]); target.eval()
    AGG="min"
    if os.environ.get("RUNJSON"):
        AGG=json.load(open(os.environ["RUNJSON"])).get("config",{}).get("actor_q_agg","min")
    res["actor_q_agg"]=AGG
    residual=None
    if "residual" in R6:
        RESP=_load("g1rp",RL+"g1_piston_residual_policy.py"); _rc=R6.get("residual_cfg",{})
        residual=RESP.ResidualPolicy(feat_dim=int(_rc.get("feat_dim",HID)),hidden=tuple(_rc.get("hidden",(512,512))),
                                     r_max=float(_rc.get("r_max",0.15)),device=DEV)
        residual.load_state_dict(R6["residual"]); residual.eval(); res["residual_applied"]=True
    PERT_STD=float(os.environ.get("PERT_STD","0.15")); K=int(os.environ.get("K","8"))

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
        with torch.no_grad():
            qa=net(feats,ABAS.flatten(ABAS.project(a)))
            return qa.mean(dim=1) if AGG=="mean" else qa.min(dim=1)[0]

    z=np.load("/home/jren313/research/starvla_rl/demo_buffer_v3/ep046.npz")
    rows=[]; feats=[]; aqs=[]
    for i in range(len(z["actions"])-1):
        aq,pooled=encode(z["images"][i].astype(np.uint8))
        cs=torch.as_tensor(z["critic_state"][i],dtype=torch.float32,device=DEV).view(1,-1)
        feats.append(torch.cat([pooled,cs],-1)); aqs.append(aq)
    F_=torch.cat(feats); AQ=torch.cat(aqs)
    a_demo=nrm.normalize(torch.as_tensor(z["actions"][:len(aqs)],dtype=torch.float32)).to(DEV)
    model.action_model.load_state_dict(BC);  a_bc=squash(head(AQ))
    # the checkpoint's own policy chunk: its head (frozen = BC under the residual arm) + residual
    model.action_model.load_state_dict(R6["action_model"]); a_pol=squash(head(AQ))
    if residual is not None:
        with torch.no_grad(): a_pol,_,_=residual.act(AQ,a_pol,None,am,deterministic=True)
        a_pol=torch.where(am,a_pol,FROZEN.expand_as(a_pol))
    torch.manual_seed(0)
    perts=[torch.where(am,torch.clamp(a_bc+PERT_STD*torch.randn_like(a_bc),LOW,HIGH),a_bc) for _ in range(K)]
    out={}
    for name,net in (("critic",critic),("target",target)):
        qd,qb,qp=q(net,F_,a_demo),q(net,F_,a_bc),q(net,F_,a_pol)
        qpert=torch.stack([q(net,F_,x) for x in perts])                     # [K,n]
        frac_bc_over_pert=float((qb.unsqueeze(0)>qpert).float().mean())
        gap=float((qb-qp).abs().mean()/qb.abs().mean().clamp(min=1e-6))
        line=[float(q(net,F_,a_bc+t*(a_pol-a_bc)).mean()) for t in (0.0,0.25,0.5,0.75,1.0)]
        out[name]={"Q_demo":float(qd.mean()),"Q_bc":float(qb.mean()),"Q_policy":float(qp.mean()),
                   "Q_pert_mean":float(qpert.mean()),
                   "frac_Q_bc_gt_Q_pert":frac_bc_over_pert,
                   "rel_gap_bc_vs_policy":gap,
                   "frac_frames_Q_policy_gt_Q_bc":float((qp>qb).float().mean()),
                   "Q_along_bc_to_policy":line,
                   "H10_sensitivity":frac_bc_over_pert>=0.70,"H10_gap":gap>0.10}
    res["n_frames"]=int(F_.shape[0]); res["by_net"]=out
    T_=torch.as_tensor(z["actions"][:len(aqs)])
    res["deployed_err_deg"]={"bc_vs_demo":float(((nrm.denormalize(a_bc.cpu())-T_).abs()[...,am.cpu()]).mean())*57.3,
                             "policy_vs_demo":float(((nrm.denormalize(a_pol.cpu())-T_).abs()[...,am.cpu()]).mean())*57.3,
                             "pert_vs_demo_mean":float(sum(((nrm.denormalize(x.cpu())-T_).abs()[...,am.cpu()]).mean() for x in perts)/K)*57.3}
    res["pert_std"]=PERT_STD; res["K"]=K
    res["environment_truth"]={"bc":"grasp 1.00, lift 0.80, return 11.935 (scorer of record)","demo_ep_returns_mean":11.88}
    emit("OK"); os._exit(0)
except Exception as e:
    res["error"]=f"{type(e).__name__}: {e}"; res["tb"]=traceback.format_exc(); emit("ERROR"); os._exit(1)
