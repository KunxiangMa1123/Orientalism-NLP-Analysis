# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 23:28:50 2025

@author: 12549
"""

import os
os.environ["WANDB_DISABLED"] = "true"  # ⛔ 禁用 Weights & Biases（无需卸载）

import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# ==== Step 1: Load and Prepare Data ====
data_path = r"C:\Users\12549\Desktop\Advanced项目\orientalism_training_data_full.xlsx"
df = pd.read_excel(data_path)
df = df[['text', 'label']].dropna()
df['label'] = df['label'].astype(int)

# Split into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'], df['label'], test_size=0.2, stratify=df['label'], random_state=42
)

# ==== Step 2: Tokenize ====
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True, max_length=256)

train_dataset = Dataset.from_dict({'text': train_texts.tolist(), 'label': train_labels.tolist()})
val_dataset = Dataset.from_dict({'text': val_texts.tolist(), 'label': val_labels.tolist()})

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)

# ==== Step 3: Load Pretrained BERT Model ====
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# ==== Step 4: Define Training Arguments ====
training_args = TrainingArguments(
    output_dir=r"C:\Users\12549\Desktop\Advanced项目\orientalism_bert_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=4,
    weight_decay=0.01,
    logging_dir=r"C:\Users\12549\Desktop\Advanced项目\logs",
    logging_steps=10,
    save_total_limit=1,
    load_best_model_at_end=True,
)

# ==== Step 5: Initialize Trainer ====
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

# ==== Step 6: Train ====
trainer.train()

# ==== Step 7: Save Final Model ====
model.save_pretrained(r"C:\Users\12549\Desktop\Advanced项目\orientalism_bert_model")
tokenizer.save_pretrained(r"C:\Users\12549\Desktop\Advanced项目\orientalism_bert_model")

print("✅ 模型训练完成并已保存。")
