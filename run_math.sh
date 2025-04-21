export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export TORCH_NCCL_TIMEOUT=1800
export VLLM_USE_V1=0
export TOKENIZERS_PARALLELISM=false

source ~/venv/rlhf/bin/activate

set -x

# TextGrad + reinforce on Math

ray start --head --node-ip-address 0.0.0.0 --num-gpus 7 --num-cpus 16 --dashboard-port 8265

# python3 -m openrlhf.textgrad.train_ppo_ray \

ray job submit --address="http://127.0.0.1:8265" \
   -- python3 -m openrlhf.textgrad.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 1 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 1 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 2 \
   --vllm_num_engines 2 \
   --vllm_tensor_parallel_size 1 \
   --feedback_vllm_num_engines 2 \
   --feedback_vllm_tensor_parallel_size 1 \
   --dataset_name math \
   --pretrain Qwen/Qwen2.5-3B-Instruct \
   --feedback_model Qwen/Qwen2.5-7B-Instruct \
   --save_path ./openrlhf/textgrad/checkpoint/qwen25-3b \
   --micro_train_batch_size 16 \
   --train_batch_size 256 \
   --micro_rollout_batch_size 32 \
   --rollout_batch_size 256 \
   --n_samples_per_prompt 1 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --max_samples 100000 \
   --generate_max_len 1024 \
   --advantage_estimator uniform \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-6 \
   --init_kl_coef 4e-2 \
   --prompt_data /home/suganuma/src/lmm-r1/openrlhf/datasets/math_data/train.jsonl \
   --prompt_eval_data /home/suganuma/src/lmm-r1/openrlhf/datasets/math_data/test.jsonl \
   --input_key problem \
   --label_key solution \
   --normalize_reward \
   --adam_offload \
   --gradient_checkpointing \
   --packing_samples \
   --save_steps -1 \
   --ckpt_path ./openrlhf/textgrad/checkpoint/qwen25-3b \
   --use_wandb 'suganuma' \
   --num_episodes 10 \
   --flash_attn \
   --eval \
   --enable_prefix_caching \
   # --use_kl_loss \
   # --kl_estimator k3 \
   # --l2 0.01 \


# You could also try
#   --use_kl_loss \
#   --kl_estimator k3 | k2 \

# also supports --advantage_estimator rloo | reinforce_baseline

ray stop --force