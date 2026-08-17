"""Does reset(seed=s) actually vary the initial scene?"""
import os,sys,json,traceback
import numpy as np
OUT=os.environ["OUTF"]; res={"resets":[]}
def emit(s="RUNNING"):
    res["_status"]=s
    with open(OUT,"w") as f: json.dump(res,f,indent=2,default=str)
try:
    os.environ.pop("DISPLAY",None)
    from isaaclab.app import AppLauncher
    app=AppLauncher(headless=True,enable_cameras=True).app
    sys.path.append("/home/jren313/miniconda3/envs/isaac/lib/python3.11/site-packages")
    sys.path.insert(0,"/home/jren313/unitree_sim_isaaclab")
    import tasks, gymnasium as gym, torch
    from isaaclab_tasks.utils import load_cfg_from_registry
    TID="Isaac-PickPlace-Piston-G129-Inspire-Joint"
    cfg=load_cfg_from_registry(TID,"env_cfg_entry_point"); cfg.seed=0; cfg.scene.num_envs=1
    cfg.scene.left_wrist_camera=None; cfg.scene.right_wrist_camera=None
    env=gym.make(TID,cfg=cfg,render_mode="rgb_array").unwrapped; sc=env.scene
    for s in [0,1,2,3,4,0]:
        env.reset(seed=s)
        bar=sc["object"].data.body_pos_w[0,1].cpu().numpy()
        pot=sc["pot"].data.root_pos_w[0].cpu().numpy()
        tube=sc["tube"].data.root_pos_w[0].cpu().numpy()
        q=sc["robot"].data.joint_pos[0].cpu().numpy()
        res["resets"].append({"seed":int(s),
            "barrel":[round(float(x),5) for x in bar],
            "pot":[round(float(x),5) for x in pot],
            "tube":[round(float(x),5) for x in tube],
            "q_sum":round(float(q.sum()),6)})
        emit()
    b=[r["barrel"] for r in res["resets"]]
    res["barrel_all_identical"]=all(x==b[0] for x in b)
    res["n_distinct_barrel"]=len({tuple(x) for x in b})
    emit("OK"); os._exit(0)
except Exception as e:
    res["error"]=f"{type(e).__name__}: {e}"; res["tb"]=traceback.format_exc(); emit("FAIL"); os._exit(1)
