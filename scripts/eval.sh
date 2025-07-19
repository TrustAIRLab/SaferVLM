#!/bin/bash

DIR=
cd $DIR
bash setup.sh
eval "$(conda shell.bash hook)"
conda activate llava

export LD_LIBRARY_PATH=""
MODEL_DIR=$DIR/checkpoints/rlhf

for dataset_name in UnsafeConcepts_TEST MME LLaVABench SMID NSFW
do
    python $DIR/eval_rlhf.py \
        --dataset_name $dataset_name \
        --save_dir $DIR/outputs/original
done


for training_type in sft dpo ppo
do
    for dataset_name in UnsafeConcepts_TEST MME LLaVABench SMID NSFW
    do
        python $DIR/eval_rlhf.py \
            --dataset_name $dataset_name \
            --lora_path $MODEL_DIR/$training_type \
            --save_dir $DIR/outputs/$training_type
    done
done