from datasets import load_dataset

def load_and_prepare():
    dataset=load_dataset("mlabonne/guanaco-llama2-1k", split="train")
    return dataset.train_test_split(test_size=0.05)