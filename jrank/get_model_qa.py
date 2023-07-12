# Adapted from FastChat implementation
# https://github.com/lm-sys/FastChat/blob/main/fastchat/eval/get_model_answer.py
# and
# https://huggingface.co/spaces/izumi-lab/stormy-7b-10ep/blob/main/app.py

import json
import os
import sys
from typing import Optional

import shortuuid
import torch
from fastchat.conversation import Conversation, SeparatorStyle, get_conv_template
from fastchat.model.model_adapter import load_model, model_adapters
from adapters import FastTokenizerAvailableBaseAdapter, RwkvWorldAdapter

from fire import Fire
from peft import PeftModel
from tqdm import tqdm
from transformers import GenerationConfig
from utils import load_jsonl, save_jsonl

model_adapters[-1] = FastTokenizerAvailableBaseAdapter()

for i in range(len(model_adapters)):
    if 'Rwkv' in type(model_adapters[i]).__name__ :
        model_adapters[i] = RwkvWorldAdapter()

def get_conv_from_template_path(template_path):
    with open(template_path, "r") as file:
        config = json.load(file)

    # Convert sep_style from string to SeparatorStyle enum
    if "sep_style" in config:
        config["sep_style"] = SeparatorStyle[config["sep_style"]]

    # Start a conversation 
    if "messages" not in config:
        config["messages"] = []

    return Conversation(**config)


@torch.inference_mode()
def get_model_answers(
    model_path: str,
    model_id,
    question_file,
    answer_file,
    # model parameters
    lora_path: Optional[str] = None,
    conv_template: Optional[str] = None,
    device: str = "cuda",
    num_gpus: int = 1,
    max_gpu_memory: Optional[str] = None,  # only relevant for numgpus > 1
    load_8bit: bool = False,
    cpu_offloading: bool = False,
    debug: bool = False,
    # generation parameters
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: float = 0,
    repetition_penalty: float = 1.0,
    num_beams: int = 1,
    max_new_tokens: int = 128,
    # just generate the prompts (for debug)
    generate_prompts: bool = False,
):
    question_jsons = load_jsonl(question_file)

    if not model_id:
        model_id = shortuuid.uuid()

    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        repetition_penalty=repetition_penalty,
    )
    # Model
    if not generate_prompts:
        model, tokenizer = load_model(
            model_path=model_path,
            device=device,
            num_gpus=num_gpus,
            max_gpu_memory=max_gpu_memory,
            load_8bit=load_8bit,
            cpu_offloading=cpu_offloading,
            debug=debug,
        )
        if lora_path is not None:
            model = PeftModel.from_pretrained(
                model, lora_path, torch_dtype=torch.float16
            )

    answer_jsons = []
    for i, ques_json in enumerate(tqdm(question_jsons)):
        idx = ques_json["question_id"]
        if os.path.exists(conv_template):
            conv = get_conv_from_template_path(conv_template)
        else:
            conv = get_conv_template(conv_template)

        conv.append_message(conv.roles[0], ques_json["text"])
        conv.append_message(conv.roles[1], None)

        # if we were doing an OA prompter/assistant conversation
        # for parent in ques_json["parents"][::-1]:
        #     if parent['role'] == 'prompter':
        #         conv.append_message(conv.roles[0], parent['text'])
        #     elif parent['role'] == 'assistant':
        #         conv.append_message(conv.roles[1], parent['text'])
        # conv.append_message(conv.roles[1], None)

        prompt = conv.get_prompt()

        if not generate_prompts:
            if 'RWKV' in model_path:
                #https://github.com/BlinkDL/ChatRWKV/blob/main/API_DEMO_WORLD.py

                input_ids = torch.Tensor(tokenizer.encode(
                    prompt
                )).unsqueeze(0)

                output_ids = model.generate(
                    input_ids=input_ids,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                )

                output_ids = output_ids[0][len(input_ids[0]) :]
                outputs = tokenizer.decode(output_ids).strip()
            else:
                input_ids = tokenizer.encode(
                    prompt, return_tensors="pt", add_special_tokens=False
                )

                output_ids = model.generate(
                    input_ids=input_ids.to(model.device),
                    generation_config=generation_config,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )


                output_ids = output_ids[0][len(input_ids[0]) :]
                outputs = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

            print(f"input_ids: {input_ids}", file=sys.stderr)
            print(f"len(input_ids): {len(input_ids)}", file=sys.stderr)
            
            print(f"outputs: {outputs}", file=sys.stderr)
            print(f"len(outputs_ids): {output_ids}", file=sys.stderr)
        else:
            outputs = ""

        answer_jsons.append(
            {
                "question_id": idx,
                "prompt": prompt,
                "text": outputs,
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "metadata": {},
            }
        )

    print(answer_jsons)
    save_jsonl(answer_jsons, answer_file)

    return answer_jsons


if __name__ == "__main__":
    Fire(get_model_answers)
