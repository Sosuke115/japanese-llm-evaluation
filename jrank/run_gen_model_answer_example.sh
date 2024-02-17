#!/bin/bash

model_path="/home/sosuke/qlora_ja/results/japanese-stablelm-base-gamma-7b-8bit-all-qlora-sft/checkpoint-5000"

python gen_model_answer.py \
    --model-path $model_path \
    --model-id stablelm-gamma-example \
    --bench-name rakuda_v2 \
    --conv_template ./templates/mistral.json \
    --load_8bit \
    --cpu_offloading \
    --max_tokens 1024 \
    --num_gpus 4 \
    --debug
