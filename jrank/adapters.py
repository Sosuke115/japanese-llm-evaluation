from fastchat.model.model_adapter import BaseModelAdapter
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import torch

# For Rinna support
class FastTokenizerAvailableBaseAdapter(BaseModelAdapter):
    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        print(
            "Loading using default adapter with model kwargs:", from_pretrained_kwargs
        )
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        except ValueError:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, **from_pretrained_kwargs
        )
        return model, tokenizer


# For JapaneseStableLM-alpha support
class JapaneseStableLMAlphaAdapter(BaseModelAdapter):
    def match(self, model_path: str):
        return (model_path.split('/')[-1] == "japanese-stablelm-instruct-alpha-7b")

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        print("Loading using Japanese-StableLM alpha adapter", file=sys.stderr)
        print("model kwargs:", from_pretrained_kwargs, file=sys.stderr)
        from transformers import LlamaTokenizer

        tokenizer = LlamaTokenizer.from_pretrained("novelai/nerdstash-tokenizer-v1", additional_special_tokens=['▁▁'])

        model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, **from_pretrained_kwargs
        )
        model.half()
        model.eval()


        return model, tokenizer

# For JapaneseStableLM-alpha-v2 support
class JapaneseStableLMAlphaAdapterv2(BaseModelAdapter):
    def match(self, model_path: str):
        match = (model_path == "japanese-stablelm-instruct-alpha-7b-v2")
        return match

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        print("Loading using Japanese-StableLM alpha v2 adapter", file=sys.stderr)
        print("model kwargs:", from_pretrained_kwargs, file=sys.stderr)
        from transformers import LlamaTokenizer

        tokenizer = LlamaTokenizer.from_pretrained("novelai/nerdstash-tokenizer-v1", additional_special_tokens=['▁▁'])

        model = AutoModelForCausalLM.from_pretrained(
            "stabilityai/japanese-stablelm-instruct-alpha-7b-v2",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            variant="fp16",
        )
        model.eval()

        return model, tokenizer
    
# For Elyza support
class ElyzaJapaneseLlama2Adapter(BaseModelAdapter):
    def match(self, model_path: str):
        return "elyza" in model_path.lower()
    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        model, tokenizer = super().load_model(model_path, from_pretrained_kwargs)
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        return model, tokenizer

# For Stablelm gammma support
class JapaneseStableLMGammaAdapter(BaseModelAdapter):
    def match(self, model_path: str):
        return "japanese-stablelm-base-gamma-7b" in model_path.lower()
    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        model, tokenizer = super().load_model(model_path, from_pretrained_kwargs)
        return model, tokenizer
