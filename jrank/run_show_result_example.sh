#!/bin/bash

python show_result.py \
    --bench-name rakuda_v2 \
    --model-list stablelm-gamma-checkpoint-3000 stablelm-gamma-checkpoint-4000 \
    --input-file data/rakuda_v2/model_judgment/gpt-4_single.jsonl
