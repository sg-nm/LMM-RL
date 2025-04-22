export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export TORCH_NCCL_TIMEOUT=1800
export VLLM_USE_V1=0

source .env
echo "HF_TOKEN is: $HF_TOKEN"

source ~/venv/rlhf/bin/activate

set -x

# TextGrad + reinforce on Card Game

export ACTOR_NUM_GPUS=4
export BATCH_SIZE_PER_GPU=4
export GRAD_ACCUM_STEPS=16
# export GLOBAL_BATCH_SIZE=$((ACTOR_NUM_GPUS * BATCH_SIZE_PER_GPU * GRAD_ACCUM_STEPS))
export GLOBAL_BATCH_SIZE=$((ACTOR_NUM_GPUS * BATCH_SIZE_PER_GPU))

# python3 -m openrlhf.textgrad.train_tg_card \

ray start --head --node-ip-address 0.0.0.0 --num-gpus 8 --num-cpus 32 --dashboard-port 8265

ray job submit --address="http://127.0.0.1:8265" \
   -- python3 -m openrlhf.textgrad.train_tg_card \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node $ACTOR_NUM_GPUS \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 1 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node $ACTOR_NUM_GPUS \
   --vllm_num_engines 2 \
   --vllm_tensor_parallel_size 1 \
   --feedback_vllm_num_engines 2 \
   --feedback_vllm_tensor_parallel_size 1 \
   --colocate_actor_ref \
   --vllm_sync_backend nccl \
   --vllm_gpu_memory_utilization 0.8 \
   --multimodal \
   --pretrain /home/suganuma/src/QwenVL_sft/output-7B-3/checkpoint-1250 \
   --feedback_model Qwen/Qwen2.5-14B-Instruct \
   --save_path ./openrlhf/textgrad/checkpoint/qwen25-3-7B \
   --micro_train_batch_size $BATCH_SIZE_PER_GPU \
   --train_batch_size $GLOBAL_BATCH_SIZE \
   --micro_rollout_batch_size 32 \
   --rollout_batch_size $BATCH_SIZE_PER_GPU \
   --grad_accum_steps $GRAD_ACCUM_STEPS \
   --n_samples_per_prompt 1 \
   --max_epochs 2 \
   --prompt_max_len 4800 \
   --max_samples 100000 \
   --generate_max_len 512 \
   --advantage_estimator uniform \
   --zero_stage 2 \
   --bf16 \
   --actor_learning_rate 1e-6 \
   --normalize_reward \
   --gradient_checkpointing \
   --save_steps -1 \
   --ckpt_path ./openrlhf/textgrad/checkpoint/qwen25-3-7B \
   --use_wandb 'suganuma' \
   --num_episodes 300 \
   --flash_attn \
   --l2 0.0 \
   --enable_prefix_caching \
   --env_config /home/suganuma/src/lmm-r1/card_env/gym_cards/configs/card_24.yaml \
   --eps_clip 0.2 \
   --init_kl_coef 1e-1 \
   --adam_offload \
   --eval \
   --use_kl_loss \
   --kl_estimator k3 \
   --log \
   --output_log_dir /home/suganuma/src/lmm-r1/openrlhf/textgrad/logs \
   # --freeze_vision_encoder \
   # --colocate_all_models \
   # --enforce_eager \
   # --vllm_enable_sleep \
   # --deepspeed_enable_sleep \
   # --pretrain /home/suganuma/src/QwenVL_sft/output-7B-3/checkpoint-1250 \
   # --pretrain Qwen/Qwen2.5-VL-7B-Instruct \

ray stop --force