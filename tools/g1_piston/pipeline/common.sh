# Shared helpers. The busy check is built so that no shell whose command line merely
# CONTAINS this file's text can match it: the literal prefix must be present and each
# tool name is bracketed (a regex "train_sa[c]" does not match the text "train_sa[c]").
D=/home/jren313/research/starvla_rl/runs_g1_piston
T=/home/jren313/research/starvla_rl/RLinf/tools/g1_piston
PY=$HOME/miniconda3/envs/env_isaaclab/bin/python
BC=/home/jren313/research/starvla_rl/checkpoints/g1_piston_bc_working/bc_ckpt_latest.pt
PAT="^[^ ]*bin/python tools/g1_piston/(train_sa[c]|eval_checkpoin[t]|render_rollout[s])\.py"
gpu_busy() { pgrep -f "$PAT" >/dev/null; }
gpu_free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
wait_gpu() { while gpu_busy || [ "$(gpu_free_mib)" -lt 9500 ]; do sleep 60; done; }
export MUJOCO_GL=egl PROJECT_ROOT=/home/jren313/unitree_sim_isaaclab PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
