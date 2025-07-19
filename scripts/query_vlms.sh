#!/bin/bash

DIR=
cd $DIR
bash setup.sh
eval "$(conda shell.bash hook)"
conda activate llava

capability=alignment
response_dir=data/VLM_responses

# setup environment for LLaVA series
for model_name in llava-v1.5-7b llava-v1.5-13b
do
    python $DIR/measure.py \
        --model_name $model_name \
        --capability $capability \
        --response_dir $response_dir
done

# InternLM
pip install transformers==4.33.2
pip install accelerate==0.20.3

model_name=internlm-7b
python $DIR/measure.py \
    --model_name $model_name \
    --capability $capability \
    --response_dir $response_dir

# CogVLM
pip install torch==2.6.0 torchvision==0.21.0 triton==3.2.0 accelerate==0.24.1 transformers==4.35.0 sentencepiece==0.1.99 einops==0.7.0 xformers==0.0.29.post2

model_name=cogvlm-7b
python $DIR/measure.py \
    --model_name $model_name \
    --capability $capability \
    --response_dir $response_dir

# InstructBLIP
pip install transformers==4.53.2
pip install --upgrade timm

for model_name in instructblip-7b instructblip-13b
do
    python $DIR/measure.py \
        --model_name $model_name \
        --capability $capability \
        --response_dir $response_dir
done

# Qwen2-VL
pip install -q -U google-generativeai
pip install tiktoken
pip install transformers_stream_generator
pip install qwen-vl-utils
pip install transformers==4.45.0
pip install accelerate==0.26.0

model_name=qwen-7b
python $DIR/measure.py \
    --model_name $model_name \
    --capability $capability \
    --response_dir $response_dir

# GPT-4V
pip install Pillow
pip install openai
export OPENAI_API_KEY=

model_name=gpt-4v
python $DIR/measure.py \
    --model_name $model_name \
    --capability $capability \
    --response_dir $response_dir