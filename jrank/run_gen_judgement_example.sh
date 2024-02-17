#!/bin/bash

python gen_judgment.py \
    --model-list stablelm-gamma-example \
    --parallel 2 \
    --bench-name rakuda_v2 \
    --mode single \
    --skip-confirm
