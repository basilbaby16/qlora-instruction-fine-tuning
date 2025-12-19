from transformers import TrainingArguments
from trl import SFTTrainer

from load_model import load_model
from prepare_dataset import load_and_prepare
from config import lora_config
from huggingface_hub import login
import os

def main():
    login(token=os.getenv("HF_TOKEN"))
    model, tokenizer=load_model()
    dataset=load_and_prepare()
    training_args= TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="epoch",
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        push_to_hub=True,
        hub_model_id="basilbaby16/llama3-qlora-vscode",
        report_to="none"
    )

    trainer=SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=1024
    )

    trainer.train()


if __name__=="__main__":
    main()

