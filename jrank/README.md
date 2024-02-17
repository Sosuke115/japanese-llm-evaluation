# How to evaluate models

Preparing Python Virtual Environment
```bash
$ pwd
/home/sosuke/japanese-llm-evaluation
$ source japanese-llm-evaluation/bin/activate
```

Generating Output for Target Model
```bash
$ pwd
/home/sosuke/japanese-llm-evaluation/jrank
$ cat run_gen_model_answer_example.sh
#!/bin/bash

model_path="/home/sosuke/qlora_ja/results/japanese-stablelm-base-gamma-7b-8bit-all-qlora-sft/checkpoint-5000"

python gen_model_answer.py \
    --model-path $model_path \
    --model-id stablelm-gamma-example \
    --bench-name rakuda_v2 \
    --conv_template ./templates/llama2-ja.json \
    --load_8bit \
    --cpu_offloading \
    --max_tokens 1024 \
    --num_gpus 4 \
    --debug

$ bash run_gen_model_answer_example.sh
# jrank/data/rakuda_v2/answers/stablelm-gamma-example.jsonl is generated
```

GPT-4 Based Judging
```bash
$ cat run_gen_judgement_example.sh 
#!/bin/bash

python gen_judgment.py \
    --model-list stablelm-gamma-example \
    --parallel 2 \
    --bench-name rakuda_v2 \
    --mode single \
    --skip-confirm
$ bash run_gen_judgement_example.sh 
# jrank/data/rakuda_v2/model_judgment/gpt-4_single.jsonl is generated
# Evaluations for all models are aggregated in this file
```


Evaluation Visualization
```bash
$ cat run_show_result_example.sh
#!/bin/bash

python show_result.py \
    --bench-name rakuda_v2 \
    --model-list stablelm-gamma-example stablelm-gamma-example2 \
    --input-file data/rakuda_v2/model_judgment/gpt-4_single.jsonl

$ bash run_show_result_example.sh
# Quantitative evaluation results are calculated for each model ID
Mode: single
Input file: data/rakuda_v2/model_judgment/gpt-4_single.jsonl

########## First turn ##########
                                     score
model                          turn       
stablelm-gamma-example 1      4.35
stablelm-gamma-example2 1      3.85
```
