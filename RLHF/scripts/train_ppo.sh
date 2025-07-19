#!/bin/bash

DIR=
cd $DIR
bash setup.sh
eval "$(conda shell.bash hook)"
conda activate llava
cd $DIR/RLHF

export LD_LIBRARY_PATH=""
export MASTER_ADDR=localhost
export MASTER_PORT=22045
export CUDA_VISIBLE_DEVICES=0
export DATA_DIR="${DIR}/data/rlhf"
export MODEL_DIR="${DIR}/checkpoints/rlhf/ppo"
export GPUS_PER_NODE=1
export WORLD_SIZE=$GPUS_PER_NODE
export OMP_NUM_THREADS=1
export TRANSFORMERS_OFFLINE=0
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# MODEL CONFIG
POLICY_BASE_MODEL_NAME=liuhaotian/llava-v1.5-7b

RM_MODEL_BASE=roberta-base
RM_MODEL_PATH=$DIR/checkpoints/alignment_classifier/epoch_10.pt

# TRAINING CONFIG
LEARNING_RATE=3e-5
KL_COEF=0.1
ENTROPY_COEF=0.02
LENGTH_BONUS=4.0
EPOCH=4

ROLLOUT_BATCH_SIZE=32
STEP_BATCH_SZIE=32
ROLLOUT_PER_DEVICE_BATCH_SIZE=32
REWARD_MODEL_PER_DEVICE_BATCH_SIZE=32
STEP_PER_DEVICE_BATCH_SIZE=32
NOPTEPOCHS=1

# DATA CONFIG
DATASET="ppo_train.json"
DATASET_NAME=ppo
IMAGE_ROOT=$DIR/data

# SAVE CONFIG
SAVE_NAME="ep${EPOCH}_kl_${KL_COEF}_entropy_${ENTROPY_COEF}_length_${LENGTH_BONUS}_lr${LEARNING_RATE}_${DATASET_NAME}"
MODEL_NAME=$MODEL_DIR/$SAVE_NAME

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=$GPUS_PER_NODE \
    --master-addr $MASTER_ADDR \
    --master-port $MASTER_PORT \
    finetune_lora_ppo.py \
    --do_train \
    --seed 42 \
    --step_batch_size $STEP_BATCH_SZIE \
    --step_per_device_batch_size $STEP_PER_DEVICE_BATCH_SIZE \
    --rollout_batch_size $ROLLOUT_BATCH_SIZE \
    --rollout_per_device_batch_size $ROLLOUT_PER_DEVICE_BATCH_SIZE \
    --reward_model_per_device_batch_size $REWARD_MODEL_PER_DEVICE_BATCH_SIZE \
    --base_model_name "$POLICY_BASE_MODEL_NAME" \
    --policy_model_name_or_path None \
    --reward_model_base "$RM_MODEL_BASE" \
    --reward_model_name_or_path "$RM_MODEL_PATH" \
    --learning_rate $LEARNING_RATE \
    --init_value_with_reward False \
    --warmup_steps 5 \
    --dataset_path "$DATA_DIR/$DATASET" \
    --train_splits "train" \
    --output_dir "$MODEL_NAME" \
    --total_epochs $EPOCH \
    --group_by_length False \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 100000 \
    --weight_decay 0.0 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --ddp_backend "nccl" \
    --bf16 True \
    --gradient_checkpointing True \
    --length_bonus_score $LENGTH_BONUS \
    --ddp_find_unused_parameters False \
    --resume_from_training False \
    --kl_coef $KL_COEF \
    --entropy_coef $ENTROPY_COEF \
    --max_grad_norm 1.0 \
    --whitening_async_stats "full_batch" \
    --clean_tokens_after_eos True \
    --temperature 1.0 \
    --whiten_rewards False \
    --model_max_length 2048 \
    --query_len 128 \
    --response_len 512 \
    --noptepochs $NOPTEPOCHS \
    --image_folder $IMAGE_ROOT \
    --vision_tower different \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --freeze_mm_mlp_adapter True \
    --image_aspect_ratio 'pad' \
    --lora_r 128 \
    --lora_alpha 256 \
    --lora_dropout 0.05
