#!/bin/bash

base_path="/home/sosuke/qlora_ja/results/studio-ousia-ELYZA-japanese-Llama-2-7b-8bit-all"

for step in {1000..5000..1000}
do
    checkpoint_path="$base_path/checkpoint-$step"

    python gen_model_answer.py \
        --model-path $checkpoint_path \
        --model-id "studio-ousia/elyza-checkpoint-$step" \
        --bench-name rakuda_v2 \
        --conv_template ./templates/llama2-ja.json \
        --load_8bit \
        --cpu_offloading \
        --max_tokens 1024 \
        --num_gpus 4 \
        --debug

    echo "Completed checkpoint $checkpoint"
done
