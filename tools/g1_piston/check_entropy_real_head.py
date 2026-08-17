"""Alpha/actor dynamics with the REAL OFT action head, on CPU (GPU is busy).

Loads only the action head from the checkpoint -- not the 2B VLM -- so this runs in
seconds and needs no simulator. Verifies the corrected entropy configuration behaves on
the actual head the trainer optimises.
"""
import sys, json, math, torch
sys.path.insert(0,"/home/jren313/research/starvla_rl/starVLA")
sys.path.insert(0,"/home/jren313/research/starvla_rl/RLinf")
from rlinf.models.embodiment.modules.q_head import MultiQHead
from rlinf.models.embodiment.modules.entropy_tunning import EntropyTemperature
from rlinf.models.embodiment.modules.gaussian_policy import SquashedNormal
from rlinf.envs.isaaclab.tasks.g1_piston_rl_space import (
    build_active_mask, default_target_entropy, load_frozen_values,
    FROZEN_ACTION_DIMS, TARGET_ENTROPY_STD)

CKPT="/home/jren313/research/starvla_rl/checkpoints/g1-longhorizon-oft-v1/final_model/pytorch_model.pt"
STATS="/home/jren313/research/starvla_rl/checkpoints/g1-longhorizon-oft-v1/dataset_statistics.json"
H,HID,AD,B=30,2048,30,8
LOW,HIGH=-2.2,2.2
res={}

sd=torch.load(CKPT,map_location="cpu",weights_only=False)
head_keys=[k for k in sd if k.startswith("action_model.")]
res["oft_head_tensors"]=len(head_keys)
res["oft_head_params"]=int(sum(sd[k].numel() for k in head_keys))

mask=build_active_mask(); frozen=load_frozen_values(STATS).float()
TARGET=default_target_entropy()
INIT_LOGSTD=math.log(TARGET_ENTROPY_STD)
res["target_entropy"]=TARGET; res["init_logstd"]=round(INIT_LOGSTD,4)

# Rebuild the real head module and load its weights.
from starVLA.model.modules.action_model.MLP_ActionHeader import get_action_model  # noqa
from starVLA.model.framework.VLM4A.QwenOFT import QwenOFTDefaultConfig
from starVLA.model.framework.share_tools import merge_framework_config
import omegaconf
cfg=omegaconf.OmegaConf.load("/home/jren313/research/starvla_rl/checkpoints/g1-longhorizon-oft-v1/config.yaml")
try:
    full=merge_framework_config(QwenOFTDefaultConfig,cfg)
    full.framework.action_model.action_hidden_dim=HID
    am=get_action_model(config=full)
    missing,unexp=am.load_state_dict(
        {k[len("action_model."):]:v for k,v in sd.items() if k.startswith("action_model.")},
        strict=False)
    res["head_loaded"]=True; res["missing"]=len(missing); res["unexpected"]=len(unexp)
except Exception as e:
    res["head_loaded"]=False; res["head_error"]=f"{type(e).__name__}: {e}"
    print(json.dumps(res,indent=1)); sys.exit(0)

am=am.float()
for p in am.parameters(): p.requires_grad_(True)
logstd=torch.nn.Parameter(torch.full((AD,),INIT_LOGSTD))
critic=MultiQHead(HID,AD*H,[256,256],num_q_heads=2)
ent=EntropyTemperature(initial_alpha=0.05,alpha_type="softplus")
opt_a=torch.optim.Adam(list(am.parameters())+[logstd],lr=3e-6)
opt_al=torch.optim.Adam(ent.parameters(),lr=1e-3)

def masked_lp(d,s):
    base=d.base_dist.base_dist
    parts=[]
    for t in d.transforms: parts.extend(getattr(t,"parts",[t]))
    x=s
    for t in reversed(parts): x=t.inv(x)
    pd=base.log_prob(x); y=x
    for t in parts:
        y2=t(y)
        if type(t).__name__=="TanhTransform": pd=pd-torch.log(1-y2.pow(2)+1e-7)
        else:
            sc=torch.as_tensor(t.scale,dtype=x.dtype); pd=pd-torch.log(sc.abs()).expand_as(x)
        y=y2
    return (pd*mask).sum(-1)

torch.manual_seed(0)
aqs=torch.randn(B,H,HID)*0.5; feats=torch.randn(B,HID)
steps=[]
for step in range(30):
    mean=am.predict_action(aqs).float()
    flat=mean.reshape(B*H,AD)
    std=torch.exp(logstd).view(1,-1).expand_as(flat)
    d=SquashedNormal(flat,std,low=LOW,high=HIGH)
    a=d.rsample(); lp=masked_lp(d,a).reshape(B,H)
    a=torch.where(mask,a,frozen.expand_as(a)).reshape(B,H,AD)
    logp_chunk=lp.mean(-1,keepdim=True)
    qpi=critic(feats,a.reshape(B,-1)).min(1,keepdim=True)[0]
    alpha=ent.compute_alpha().detach()
    ent_term=alpha*logp_chunk
    aloss=(ent_term-qpi).mean()
    opt_a.zero_grad(); aloss.backward()
    gn=sum(p.grad.abs().sum().item() for p in am.parameters() if p.grad is not None)
    opt_a.step()
    av=ent.compute_alpha()
    alloss=-av*(logp_chunk.mean().detach()+TARGET)
    opt_al.zero_grad(); alloss.backward(); opt_al.step()
    if step in (0,10,20,29):
        steps.append(dict(step=step,logp=round(float(logp_chunk.mean()),2),
            alpha=round(float(av),5),q=round(float(qpi.mean()),3),
            ent_term=round(float(ent_term.mean()),3),
            ratio=round(abs(float(ent_term.mean()))/max(abs(float(qpi.mean())),1e-6),2),
            head_grad=round(gn,1),
            frozen_dev=float((a[...,list(FROZEN_ACTION_DIMS)]-frozen[list(FROZEN_ACTION_DIMS)]).abs().max()),
            all_finite=bool(torch.isfinite(aloss) and torch.isfinite(alloss) and torch.isfinite(av))))
res["steps"]=steps
res["alpha_first"]=steps[0]["alpha"]; res["alpha_last"]=steps[-1]["alpha"]
res["initial_logp_vs_target"]=round(steps[0]["logp"]-TARGET,2)
print(json.dumps(res,indent=1))
