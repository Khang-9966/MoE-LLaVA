#!/bin/bash

JSON_FOLDER="/MoE-LLaVA/pt_data/json"
IMAGE_FOLDER="/MoE-LLaVA/pt_data/images"
cd /MoE-LLaVA
deepspeed moellava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path Qwen2-0.5B-1.5-sailor \
    --version plain \
    --data_path ${JSON_FOLDER}/pt_moe_llava.json \
    --image_folder ${IMAGE_FOLDER} \
    --image_tower openai/clip-vit-base-patch16 \
    --image_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/llavaqwen-1.5-0.5b-pretrain-vi-500k \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir "./cache_dir"