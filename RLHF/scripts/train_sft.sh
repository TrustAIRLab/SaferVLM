#!/bin/bash

DIR=
cd $DIR
bash setup.sh
eval "$(conda shell.bash hook)"
conda activate llava
cd $DIR/RLHF
export LD_LIBRARY_PATH=""

export MASTER_ADDR=localhost
export MASTER_PORT=22032
export CUDA_VISIBLE_DEVICES=0
export DATA_DIR="${DIR}/data/rlhf"
export MODEL_DIR="${DIR}/checkpoints/rlhf/sft"
export GPUS_PER_NODE=1
export OMP_NUM_THREADS=1

# MODEL CONFIG
VISION_TOWER=openai/clip-vit-large-patch14
LM_MODEL_NAME=liuhaotian/llava-v1.5-7b

# TRAINING CONFIG
NUM_EPOCHS=4
LEARNING_RATE=3e-5
BATCH_SIZE=32
GRAD_ACCUMULATION=1

# DATA CONFIG
DATASET="sft_train.json"
DATASET_NAME=sft_train
IMAGE_ROOT=$DIR/data

# SAVE CONFIG
SAVE_NAME="ep${EPOCH}_lr${LEARNING_RATE}_${DATASET_NAME}"
MODEL_NAME=$MODEL_DIR/$SAVE_NAME

torchrun \
    --master-addr $MASTER_ADDR \
    --master-port $MASTER_PORT \
    finetune_lora_sft.py \
    --do_train \
    --seed 42 \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUMULATION \
    --base_model_name $LM_MODEL_NAME \
    --image_folder $IMAGE_ROOT \
    --vision_tower $VISION_TOWER \
    --learning_rate $LEARNING_RATE \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --freeze_mm_mlp_adapter True \
    --query_len 128 \
    --response_len 512 \
    --dataset "$DATA_DIR/$DATASET" \
    --dataset_format "v1" \
    --lora_r 128 \
    --lora_alpha 256 \
    --lora_dropout 0.05 \
    --output_dir "$MODEL_NAME" \
    --num_train_epochs $NUM_EPOCHS \
    --group_by_length False \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 10 \
    --weight_decay 0.0 \
    --warmup_steps 5 \
    --lr_scheduler_type "cosine" \
    --logging_steps 50 \
    --ddp_backend "nccl" \
    --bf16 True \
    --gradient_checkpointing True \
    --ddp_find_unused_parameters False \
    --resume_from_training False \
    --image_aspect_ratio 'pad'
