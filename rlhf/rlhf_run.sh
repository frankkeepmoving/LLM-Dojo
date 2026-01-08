# 使用显卡数量需在yaml文件中修改num_processes参数

# rlhf_type:[PPO,RLOO,CPO,DPO,SimPO,CPOSimPO,Reward]
# train_mode:[lora, qlora, full]

TRAIN_DATA='/data/code/LLM-Dojo/llm_tricks/DPO_example/unsloth_dpo.jsonl'
MODEL_PATH='/data/models/qwen2.5-7b'
OUTPUT_PATH='./output'

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file ./ds_config/ds_zero2.yaml ./train_rlhf.py \
    --model_name_or_path "$MODEL_PATH" \
    --train_data_path "$TRAIN_DATA" \
    --output_dir "$OUTPUT_PATH" \
    --rlhf_type "DPO" \
    --train_mode "lora" \
    --learning_rate 5e-5 \
    --per_device_train_batch_size 1 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 4 \
    --logging_steps 2 \
    --num_train_epochs 3 \
    --bf16 \
    --save_strategy "steps" \
    --report_to "tensorboard" \
    --save_steps 180 \
    --save_total_limit 5 \
    --warmup_ratio 0.03 \
    --remove_unused_columns False\
    --loss_type "robust" \
    --beta 0.3 \
    --label_smoothing 0.1 \
    --lr_scheduler_type "cosine"

# [CPO,DPO,SimPO,CPOSimPO,Reward] 可直接使用上述运行

# [PPO,RLOO] 需要额外添加如下参数：
# --reward_model_path './'\
# --local_rollout_forward_batch_size 1\
# --missing_eos_penalty 1.0\
# --num_ppo_epochs 1 \
# --num_mini_batches 1