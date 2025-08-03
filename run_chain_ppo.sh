#!/usr/bin/env bash
set -e



export RM_API_BASE_URL=""
export RM_API_KEY=""

export RAY_ENABLE_BACKWARD_COMPATIBILITY=0
export RAY_IGNORE_UNHANDLED_ERRORS=0
export PYTHONUNBUFFERED=1
export VLLM_LOGGING_LEVEL=DEBUG 

export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_V1=1


ray start --head --num-gpus 2

python -m openrlhf.cli.train_ppo_ray \
  --actor_num_nodes 1 \
  --actor_num_gpus_per_node 2 \
  --ref_num_nodes 1 \
  --ref_num_gpus_per_node 2 \
  --agent_func_path /path/to/mas-train.py \
  --vllm_num_engines 2 \
  --vllm_tensor_parallel_size 1 \
  --colocate_all_models \
  --vllm_gpu_memory_utilization 0.7 \
  --pretrain /path/to/pretrain_model \
  --prompt_data /path/to/data/train/rl_data.jsonl \
  --input_key prompt \
  --label_key label \
  --zero_stage 3 \
  --bf16 \
  --flash_attn \
  --micro_train_batch_size 1 \
  --train_batch_size 2 \
  --rollout_batch_size 2 \
  --micro_rollout_batch_size 1 \
  --n_samples_per_prompt 2 \
  --max_epochs 1 \
  --max_samples 20000 \
  --actor_learning_rate 5e-7 \
  --init_kl_coef 1e-4 \
  --normalize_reward \
  --save_steps 180 \
  --ckpt_path /path/to/save_model \
  --save_hf_ckpt \
  --max_ckpt_num 1 \
  --adam_offload \
  --advantage_estimator group_norm \
  --use_kl_loss \



