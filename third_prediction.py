# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 18:28:19 2025

@author: 12549
"""

import os
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
from tqdm import tqdm

# ==== 模型路径 ====
model_path = r"C:/Users/12549/Desktop/Advanced项目/bert_finetuned_round3"

# ==== 输入与输出路径 ====
input_root = r"C:/Users/12549/Desktop/Advanced项目/预测结果_round2"
output_root = r"C:/Users/12549/Desktop/Advanced项目/预测结果_round3"
os.makedirs(output_root, exist_ok=True)

# ==== 输入文件列表 ====
input_files = [
    "china_mentions_1850_predicted_round2.xlsx",
    "china_mentions_1890_predicted_round2.xlsx",
    "china_mentions_1920_predicted_round2.xlsx"
]

# ==== 加载模型与分词器 ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained(model_path, local_files_only=True)
model = BertForSequenceClassification.from_pretrained(model_path, local_files_only=True)
model.to(device)
model.eval()

# ==== 开始预测 ====
for file in input_files:
    print(f"📂 正在处理文件：{file}")
    input_path = os.path.join(input_root, file)
    output_path = os.path.join(output_root, file.replace("round2", "round3"))

    df = pd.read_excel(input_path)
    df["sentence"] = df["sentence"].astype(str)
    sentences = df["sentence"].tolist()

    predictions = []
    probs = []

    with torch.no_grad():
        for sent in tqdm(sentences):
            inputs = tokenizer(sent, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            probs_tensor = softmax(outputs.logits, dim=1)
            prob_1 = probs_tensor[0][1].item()

            label = 1 if prob_1 > 0.4 else 0
            predictions.append(label)
            probs.append(prob_1)

    df["prob_1"] = probs
    df["prediction_round3_thresh_0.4"] = predictions

    df.to_excel(output_path, index=False)
    print(f"✅ 预测完成，已保存至：{output_path}\n")
