# model_name_or_path='studio-ousia/ELYZA-japanese-Llama-2-7b-8bit'
adapter_path="/home/sosuke/qlora_ja/results/studio-ousia-ELYZA-japanese-Llama-2-7b-8bit-all/checkpoint-1000"

python gen_model_answer.py \
    --model-path $adapter_path \
    --model-id "elyza-1000" \
    --bench-name rakuda_v2 \
    --conv_template llama2-ja \
    --load_8bit \
    --cpu_offloading \
    --max_tokens 1024 \
    --num_gpus 4 \
    --debug
    
