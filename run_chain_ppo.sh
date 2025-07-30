#!/usr/bin/env bash
set -e


# 你的闭源 RM 服务地址（示例）
export RM_API_BASE_URL="https://api.studio.nebius.com/v1/"
export RM_API_KEY="eyJhbGciOiJIUzI1NiIsImtpZCI6IlV6SXJWd1h0dnprLVRvdzlLZWstc0M1akptWXBvX1VaVkxUZlpnMDRlOFUiLCJ0eXAiOiJKV1QifQ.eyJzdWIiOiJnb29nbGUtb2F1dGgyfDExNjc3NTM1MDEyMDgzOTIyMzc2MCIsInNjb3BlIjoib3BlbmlkIG9mZmxpbmVfYWNjZXNzIiwiaXNzIjoiYXBpX2tleV9pc3N1ZXIiLCJhdWQiOlsiaHR0cHM6Ly9uZWJpdXMtaW5mZXJlbmNlLmV1LmF1dGgwLmNvbS9hcGkvdjIvIl0sImV4cCI6MTkxMTExNTg5NCwidXVpZCI6IjllOGY1YzQ1LWU3M2EtNDJjZS05NzYzLWVjODQ3ODE1NjgxOSIsIm5hbWUiOiJhZ2VudCIsImV4cGlyZXNfYXQiOiIyMDMwLTA3LTI0VDA5OjMxOjM0KzAwMDAifQ.9LkGVq9Li3delWpr9IJDnWbSysbrKd9WYhBc7Z5N66o"

export RAY_ENABLE_BACKWARD_COMPATIBILITY=0
export RAY_IGNORE_UNHANDLED_ERRORS=0
export PYTHONUNBUFFERED=1
export VLLM_LOGGING_LEVEL=DEBUG 

export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_V1=1

ray start --head --num-gpus 1

python -m openrlhf.cli.train_ppo_ray \
  --actor_num_nodes 1 \
  --actor_num_gpus_per_node 1 \
  --ref_num_nodes 1 \
  --ref_num_gpus_per_node 1 \
  --agent_func_path ./mas.py \
  --vllm_num_engines 1 \
  --vllm_tensor_parallel_size 1 \
  --colocate_all_models \
  --vllm_gpu_memory_utilization 0.7 \
  --pretrain /home/ubuntu/MAS-OpenRLHF/qwen1.5b \
  --prompt_data /home/ubuntu/MAS-OpenRLHF/data/rl_data.jsonl \
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
  --save_steps 100 \
  --ckpt_path /home/ubuntu/MAS-OpenRLHF/ppo_checkpoints \
  --save_hf_ckpt \
  --max_ckpt_num 2 \
  --adam_offload \
  --advantage_estimator group_norm \
  --use_kl_loss \



