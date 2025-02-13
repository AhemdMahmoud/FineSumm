# Summarization Fine-Tuning with Hugging Face Accelerate & Trainer API

## 📌 Project Overview
This project focuses on **fine-tuning a transformer-based model for text summarization** using **Hugging Face's Transformers**, **Trainer API**, and **Accelerate** for efficient training and evaluation. The model is trained on a dataset of text documents and generates concise summaries. 

### 🔥 Key Features
- Fine-tunes a **pretrained summarization model** (e.g., `T5`, `BART`, or `Pegasus`).
- Uses **Hugging Face Accelerate** for optimized multi-GPU/TPU training.
- Implements **Trainer API** for simplified training and evaluation.
- Uses **ROUGE** score for performance evaluation.
- Supports **automatic model saving and uploading to Hugging Face Hub**.
- Efficient batch processing and padding for stable training.

---

## 🛠️ Installation
Ensure you have Python installed (>=3.8) and run the following commands:
```bash
pip install transformers datasets accelerate torch tqdm rouge-score
```

For multi-GPU support, configure `accelerate`:
```bash
accelerate config
```

---

## 📂 Dataset
The dataset used in this project is the **SAMSum dataset**, which contains about **16,000** messenger-like conversations with summaries. Conversations were created by linguists fluent in English, reflecting real-life messenger interactions. The dataset includes informal, semi-formal, and formal conversations, with annotations that summarize the dialogues in third-person. 

The **SAMSum dataset** was prepared by **Samsung R&D Institute Poland** and is distributed for research purposes under the **CC BY-NC-ND 4.0** license.

### Loading the Dataset
```python
from datasets import load_dataset

dataset = load_dataset("samsum")
```

---

## 🚀 Model Training with Trainer API
Run the training script to fine-tune the model using **Trainer API**:
```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
import torch

# Load dataset
dataset = load_dataset("samsum")

# Load model and tokenizer
model_name = "facebook/bart-large-cnn"  # Example model
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["dialogue"], padding="max_length", truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    save_total_limit=2,
    save_steps=500,
    logging_dir="./logs",
    logging_steps=100,
    push_to_hub=True
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"]
)

# Train the model
trainer.train()
```

---

## 📊 Evaluation (ROUGE Score)
To evaluate summarization quality, we use **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**:
```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
scores = scorer.score("Generated summary text", "Reference summary text")
print(scores)
```

---

## 💾 Saving & Uploading Model
Save and upload the fine-tuned model:
```python
trainer.save_model("output_dir")
tokenizer.save_pretrained("output_dir")
trainer.push_to_hub()
```

---

## 📌 Future Improvements
- Experiment with different summarization models (`T5`, `BART`, `Pegasus`)
- Implement **beam search** or **top-k sampling** for better generation
- Hyperparameter tuning for improved performance
- Train on a custom dataset with domain-specific text

---
# Scores and Evaluation Finally
![epochs](https://github.com/AhemdMahmoud/FineSumm/blob/main/%D8%A7%D8%AE%D9%8A%D8%B1%D8%A7.png)

![Evaluation Finally](https://github.com/AhemdMahmoud/FineSumm/blob/main/evaaluation%20%D8%A7%D8%AE%D9%8A%D8%B1%D8%A7.png)



## 🤝 Contributing 
Feel free to contribute by submitting issues or pull requests. 🚀

[Click here to open the notebook in Kaggel](https://www.kaggle.com/code/mkagglec/notebook074ca7c832/edit)

[Click here to open the notebook in Google Colab](https://colab.research.google.com/drive/1vaJZIbUqucrMTia3eI1FsUWg7IivssHG#scrollTo=MQVNXQ8FCoRE)

---

## 📜 License
This project is licensed under the MIT License.

---

## ✨ Acknowledgments
Thanks to **Hugging Face** for their amazing NLP tools and the **Accelerate** team for optimized training solutions.

