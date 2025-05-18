export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export TORCH_NCCL_TIMEOUT=1800
export VLLM_USE_V1=0

echo 3000 | sudo tee /proc/sys/kernel/keys/maxkeys
echo 300000 | sudo tee /proc/sys/kernel/keys/maxbytes
echo 1248576 | sudo tee /proc/sys/fs/aio-max-nr
echo 1248576 | sudo tee /proc/sys/fs/inotify/max_user_instances
echo 1248576 | sudo tee /proc/sys/fs/inotify/max_user_watches

source .env
echo "HF_TOKEN is: $HF_TOKEN"

# source ~/venv/qwen3/bin/activate

set -x

export ACTOR_NUM_GPUS=4
export BATCH_SIZE_PER_GPU=8
export GRAD_ACCUM_STEPS=4
export GLOBAL_BATCH_SIZE=$((ACTOR_NUM_GPUS * BATCH_SIZE_PER_GPU * GRAD_ACCUM_STEPS))
# export GLOBAL_BATCH_SIZE=$((ACTOR_NUM_GPUS * BATCH_SIZE_PER_GPU))

ray start --head --node-ip-address 0.0.0.0 --num-gpus 8 --num-cpus 32 --dashboard-port 8265

ray job submit --address="http://127.0.0.1:8265" \
   -- python3 -m openrlhf.textgrad.train_gui_agent \
   --multimodal \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node $ACTOR_NUM_GPUS \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 1 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node $ACTOR_NUM_GPUS \
   --vllm_num_engines $ACTOR_NUM_GPUS \
   --vllm_tensor_parallel_size 1 \
   --feedback_vllm_num_engines 4 \
   --feedback_vllm_tensor_parallel_size 1 \
   --colocate_all_models \
   --vllm_sync_backend nccl \
   --vllm_gpu_memory_utilization 0.5 \
   --pretrain ByteDance-Seed/UI-TARS-1.5-7B \
   --feedback_model Qwen/Qwen2.5-VL-7B-Instruct \
   --grounding_model ByteDance-Seed/UI-TARS-1.5-7B \
   --grounding_num_engines 1 \
   --grounding_tensor_parallel_size 1 \
   --save_path ./openrlhf/textgrad/checkpoint/qwen25-3-7B \
   --micro_train_batch_size $BATCH_SIZE_PER_GPU \
   --train_batch_size $GLOBAL_BATCH_SIZE \
   --micro_rollout_batch_size 32 \
   --rollout_batch_size $BATCH_SIZE_PER_GPU \
   --grad_accum_steps $GRAD_ACCUM_STEPS \
   --n_samples_per_prompt 1 \
   --max_epochs 2 \
   --prompt_max_len 4200 \
   --max_samples 100000 \
   --generate_max_len 512 \
   --advantage_estimator uniform \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --normalize_reward \
   --gradient_checkpointing \
   --save_steps 50 \
   --ckpt_path ./openrlhf/textgrad/checkpoint/qwen25-3-7B \
   --use_wandb 'suganuma' \
   --num_episodes 300 \
   --flash_attn \
   --l2 0.001 \
   --enable_prefix_caching \
   --env_config /home/suganuma/src/lmm-r1_L/LMM-RL-GUI/card_env/gym_cards/configs/card_24.yaml \
   --eps_clip 0.2 \
   --init_kl_coef 5e-2 \
   --adam_offload \
   --use_kl_loss \
   --kl_estimator k3 \
   --log \
   --output_log_dir /home/suganuma/src/lmm-r1_L/LMM-RL-GUI/openrlhf/textgrad/logs \
   --colocate_actor_vllm \
   --vllm_enable_sleep \
   --enforce_eager \
   --gamma 1.0 \
   --seed 128 \
   --use_reward_diff \
   --reasoning_logprob_weight 1.0 \
   --delta_coef 0.9 \
   --reward_coef 0.1 \
   --vm_config /home/suganuma/src/lmm-r1_L/LMM-RL-GUI/gui_env/configs/vmconfig.yaml \
   # --eval \
   # --distillation \
   # --distillation_coef 0.5 \
   # --multimodal \
   # --deepspeed_enable_sleep \
   # --eval \
   # --freeze_vision_encoder \
   # --colocate_all_models \
   # --enforce_eager \
   # --vllm_enable_sleep \
   # --deepspeed_enable_sleep \
   # --pretrain Qwen/Qwen2.5-7B-Instruct \

ray stop --force