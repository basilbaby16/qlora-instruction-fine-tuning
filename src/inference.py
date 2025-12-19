from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
ADAPTER="basilbaby16/llama3-qlora-vscode"

tokenizer=AutoTokenizer.from_pretrained(BASE_MODEL)

model=AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    load_in_4bit=True
)

model=PeftModel.from_pretrained(model,ADAPTER)

prompt="Explain LoRA in simple terms."
inputs=tokenizer(prompt,return_tensors="pt").to("cuda")

outputs=model.generate(**inputs, max_new_tokens=150)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))