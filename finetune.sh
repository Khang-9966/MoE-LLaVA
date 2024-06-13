#!/bin/bash

JSON_FOLDER="/MoE-LLaVA/ft_data/json"
IMAGE_FOLDER="/MoE-LLaVA/ft_data/images/vi_wit/images/"
cd /MoE-LLaVA
deepspeed moellava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path Qwen2-0.5B-1.5-sailor \
    --version qwen \
    --data_path ${JSON_FOLDER}/ft_moe_llava.json \
    --image_folder ${IMAGE_FOLDER} \
    --image_tower openai/clip-vit-base-patch16 \
    --image_projector_type mlp2x_gelu \
    --pretrain_mm_mlp_adapter ./checkpoints/llavaqwen-1.5-0.5b-pretrain-llava-laion-500k/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llavaqwen-1.5-0.5b-finetune-new/ \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 3 \
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
