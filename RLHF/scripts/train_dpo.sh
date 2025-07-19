#!/bin/bash

DIR=
cd $DIR
bash setup.sh
eval "$(conda shell.bash hook)"
conda activate llava
cd $DIR/RLHF

export LD_LIBRARY_PATH=""
export MASTER_ADDR=localhost
export MASTER_PORT=22003
export CUDA_VISIBLE_DEVICES=0
export DATA_DIR="${DIR}/data/rlhf"
export MODEL_DIR="${DIR}/checkpoints/rlhf/dpo"
export GPUS_PER_NODE=1
export OMP_NUM_THREADS=1
export TRANSFORMERS_OFFLINE=0

# MODEL CONFIG
VISION_TOWER=openai/clip-vit-large-patch14
LM_MODEL_NAME=liuhaotian/llava-v1.5-7b

# TRAINING CONFIG
LEARNING_RATE=2e-6
BETA=0.1
EPOCH=4
PER_DEVICE_BATCH_SIZE=32

# DATA CONFIG
DATASET=dpo_train.json
DATASET_NAME=dpo_train
IMAGE_ROOT=$DIR/data

# SAVE CONFIG
SAVE_NAME="ep${EPOCH}_lr${LEARNING_RATE}_beta_${BETA}_${DATASET_NAME}"
MODEL_NAME=$MODEL_DIR/$SAVE_NAME

torchrun \
    --master-addr $MASTER_ADDR \
    --master-port $MASTER_PORT \
    finetune_lora_dpo.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --lora_dropout 0.05 --mm_projector_lr 0 \
    --model_name_or_path $LM_MODEL_NAME \
    --version v1 \
    --dataset_path "$DATA_DIR/$DATASET" \
    --image_folder $IMAGE_ROOT \
    --vision_tower $VISION_TOWER \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir $MODEL_NAME \
    --num_train_epochs $EPOCH \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEVICE_BATCH_SIZE  \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 10 \
    --learning_rate $LEARNING_RATE \
    --weight_decay 0.0 \
    --warmup_steps 5 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 False \
    --fp16 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --beta $BETA