# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 15:29:02 2025

@author: 12549
"""

import pandas as pd
import os

# 文件路径设置（输入和输出）
base_path = r"C:/Users/12549/Desktop/Advanced项目/预测结果_round3"

# 要处理的时间段标签
time_tags = ["1850", "1890", "1920"]

# 批量处理
for tag in time_tags:
    input_file = os.path.join(base_path, f"china_mentions_{tag}_ensemble_infile.xlsx")
    output_file = os.path.join(base_path, f"china_mentions_{tag}_final_filtered.xlsx")

    # 加载文件
    df = pd.read_excel(input_file)

    # 筛选 final == 1 并保留必要字段
    filtered_df = df[df["final"] == 1][["sentence", "final"]]

    # 保存为新文件
    filtered_df.to_excel(output_file, index=False)
    print(f"✅ 已处理：{tag}，结果保存为：{output_file}")
