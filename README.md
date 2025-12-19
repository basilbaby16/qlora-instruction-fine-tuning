# 🚀 QLoRA Instruction Fine-Tuning (PEFT)

This repository demonstrates **instruction fine-tuning of a Large Language Model (LLM)** using **QLoRA (Quantized Low-Rank Adaptation)** and **PEFT**, following modern best practices from Hugging Face.

The project is designed as a **portfolio-ready, end-to-end example**, starting from dataset preparation to training on GPU (Google Colab) and deployment-ready adapters.

---

## 📌 Project Highlights

* ✅ Parameter-Efficient Fine-Tuning (PEFT)
* ✅ 4-bit quantization with **QLoRA** (memory efficient)
* ✅ Uses Hugging Face **transformers**, **peft**, **trl**, and **bitsandbytes**
* ✅ Training compatible with **Google Colab (T4 GPU)**
* ✅ Clean, modular Python project structure

---

## 🧠 What is QLoRA?

QLoRA allows fine-tuning very large models (7B–13B+) on consumer GPUs by:

* Loading the base model in **4-bit NF4 quantization**
* Freezing base weights
* Training only small **LoRA adapters**

This reduces memory usage from **~40GB → ~8GB VRAM**.

---

## 🏗️ Project Structure

```text
qlora-instruction-finetuning/
│
├── src/
│   ├── train.py              # Main training script
│   ├── load_model.py         # Loads 4-bit quantized model
│   ├── prepare_dataset.py    # Dataset loading & formatting
│   └── config.py             # Model & training config
│
├── requirements.txt          # Python dependencies
├── .gitignore
└── README.md
```

---

## 🧩 Base Model

You can use any open-source causal LLM supported by Hugging Face.

Recommended:

* `mistralai/Mistral-7B-Instruct-v0.2`
* `meta-llama/Meta-Llama-3-8B-Instruct` *(requires license acceptance)*

> ⚠️ Training **must be done on GPU** (Google Colab / Kaggle / Cloud VM)

---

## 📦 Installation (Google Colab)

```bash
pip install -r requirements.txt
```

If needed:

```bash
pip install bitsandbytes
```

---

## 🔐 Hugging Face Authentication

Some models are gated and require authentication.

```python
from huggingface_hub import login
login()
```

Paste your Hugging Face access token when prompted.

---

## 📚 Dataset Format

The dataset must be formatted as **instruction-style conversations**.

Example:

```text
<|system|>
You are a helpful assistant.
<|user|>
Summarize the following text...
<|assistant|>
Here is the summary...
```

The training script expects a column named:

```python
text
```

---

## 🏋️ Training

Run training with:

```bash
python src/train.py
```

Key training features:

* QLoRA (4-bit NF4)
* LoRA adapters (trainable parameters only)
* Optimized for low VRAM usage

---

## 💾 Output

The training process saves:

* LoRA adapter weights
* Trainer checkpoints

These adapters can be:

* Re-loaded with the base model
* Pushed to Hugging Face Hub
* Used in inference or demos

---

## 🌐 Running on Google Colab

1. Push this repo to GitHub
2. Open Google Colab
3. Enable **GPU** (`Runtime → Change runtime type`)
4. Clone repo:

```python
!git clone https://github.com/basilbaby16/qlora-instruction-finetuning.git
%cd qlora-instruction-finetuning
```

5. Install dependencies
6. Run training

---

## 🚀 Future Improvements

* [ ] Add Gradio demo
* [ ] Push adapters to Hugging Face Hub
* [ ] Experiment with different LoRA ranks
* [ ] Add evaluation metrics

---

## ⭐ Acknowledgements

* Hugging Face 🤗
* QLoRA paper by Dettmers et al.
* PEFT & TRL libraries

---

## 📜 License

This project is for **educational and research purposes**.
Model licenses follow their respective Hugging Face terms.
