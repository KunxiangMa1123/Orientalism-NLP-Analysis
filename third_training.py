# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 17:53:07 2025

@author: 12549
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 不使用GPU可手动设空，如需GPU可注释掉

import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset

# ==== 1. 路径设置 ====
train_file = r"C:/Users/12549/Desktop/Advanced项目/third.xlsx"  # 第三轮标注后的数据
pretrained_model_path = os.path.abspath("C:/Users/12549/Desktop/Advanced项目/bert_finetuned_round2")  # 第二轮模型
output_dir = r"C:/Users/12549/Desktop/Advanced项目/bert_finetuned_round3"  # 保存第三轮模型的路径
os.makedirs(output_dir, exist_ok=True)

# ==== 2. 加载数据 ====
df = pd.read_excel(train_file)
df = df.rename(columns={"prediction": "label"})  # 确保列名为 label
df["label"] = df["label"].astype(int)
df["sentence"] = df["sentence"].astype(str)

# ==== 3. 划分训练/验证集 ====
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["sentence"].tolist(),
    df["label"].tolist(),
    test_size=0.2,
    random_state=42
)

# ==== 4. 分词器和数据集处理 ====
tokenizer = BertTokenizer.from_pretrained(pretrained_model_path, local_files_only=True)

def tokenize(example):
    return tokenizer(example["sentence"], padding="max_length", truncation=True, max_length=256)

train_dataset = Dataset.from_dict({"sentence": train_texts, "label": train_labels}).map(tokenize, batched=True)
val_dataset = Dataset.from_dict({"sentence": val_texts, "label": val_labels}).map(tokenize, batched=True)

# ==== 5. 加载上轮训练的模型 ====
model = BertForSequenceClassification.from_pretrained(pretrained_model_path, num_labels=2, local_files_only=True)

# ==== 6. 设置训练参数 ====
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir=os.path.join(output_dir, "logs"),
    logging_steps=10,
    load_best_model_at_end=True,
    save_total_limit=2,
    report_to=[]  # 禁用 wandb
)

# ==== 7. 初始化 Trainer ====
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

# ==== 8. 启动训练 ====
trainer.train()
trainer.save_model(output_dir)

print(f"✅ 第三轮微调训练完成，模型已保存至：{output_dir}")
