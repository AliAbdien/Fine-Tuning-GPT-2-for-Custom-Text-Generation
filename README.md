# 🚀 Fine-Tuning GPT-2

This repository contains the code and configurations for fine-tuning **GPT-2** on a custom dataset. The model has been trained using **Hugging Face's Transformers** library, leveraging PyTorch for deep learning optimizations.

## 📂 Project Structure

```
📦 Fine-Tuning-GPT2
│── 📜 config.json          # Model configuration file
│── 📜 generation_config.json  # Configuration for text generation
│── 📜 merges.txt           # Tokenizer merge file
│── 📜 special_tokens_map.json  # Special tokens configuration
│── 📜 tokenizer_config.json   # Tokenizer settings
│── 📜 training_args.bin    # Training arguments and hyperparameters
│── 📜 vocab.json           # Tokenizer vocabulary file
│── 📜 GPT2.ipynb           # Jupyter Notebook for fine-tuning and inference
```

## 📌 Features

✅ Fine-tunes **GPT-2** on a custom dataset  
✅ Utilizes **Hugging Face Transformers** for model training  
✅ Includes model checkpoints for continued training  
✅ Supports **special tokens** and custom vocabulary  
✅ Implements **text generation** using the trained model  

## 🛠 Installation

Make sure you have Python 3.8+ installed. Then, install the required dependencies:

```bash
pip install torch transformers datasets safetensors
```

## 🚀 Fine-Tuning GPT-2

You can fine-tune the model using the provided Jupyter Notebook:

```bash
jupyter notebook GPT2.ipynb
```

Alternatively, you can run the training script in a Python environment:

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Custom training arguments
training_args = TrainingArguments(
    output_dir="./checkpoint",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    save_steps=500,
    save_total_limit=2
)

# Define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=your_dataset
)

# Start training
trainer.train()
```

## 📊 Generating Text with Fine-Tuned Model

Once the model is trained, you can use it to generate text:

```python
from transformers import pipeline

generator = pipeline("text-generation", model="./checkpoint-15500")
output = generator("Once upon a time", max_length=50)
print(output)
```

---

📧 **Contact:** ali.abdien.omar@gmail.com  

---
