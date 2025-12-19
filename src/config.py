from peft import LoraConfig

lora_config= LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","v_proj","k_proj","o_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")