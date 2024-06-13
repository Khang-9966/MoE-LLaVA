#!/bin/bash

moe_mode="sparse"
num_experts=4
top_k_experts=2
use_residual=False
router_aux_loss_coef=0.01
JSON_FOLDER="/MoE-LLaVA/ft_data/"
IMAGE_FOLDER="/MoE-LLaVA/ft_data/images"
cd /MoE-LLaVA
deepspeed moellava/train/train_mem.py \
    --moe_enable True --num_experts ${num_experts} --top_k_experts ${top_k_experts} --capacity_factor 1.5 \
    --moe_mode ${moe_mode} --use_residual ${use_residual} --router_aux_loss_coef ${router_aux_loss_coef} \
    --train_modules mlp.w1 mlp.w2 mlp.c_proj wg \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ./checkpoint-4000-qwen-1.5 \
    --version qwen \
    --data_path ${JSON_FOLDER}/ft_moe_llava.json \
    --image_folder ${IMAGE_FOLDER} \
    --image_tower openai/clip-vit-base-patch16 \
    --image_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llavaqwen-1.8b-finetune-moe \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir "./cache_dir"
