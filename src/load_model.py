import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig

MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"

def load_model():
    quant_config= BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    tokenizer=AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token=tokenizer.eos_token
    model=AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=quant_config, device_map="auto")
    model.config.use_cache=False
    return model, tokenizer