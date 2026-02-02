from transformers import LongformerTokenizer, LongformerModel
import torch
import json
import numpy as np
from tqdm import tqdm

# 从配置文件导入
from config import REMOVE_BOT_FILE, FILTERED_DIFF_FILE, TOKEN_LENGTH_DIST_FILE

model_name = 'allenai/longformer-base-4096'
tokenizer = LongformerTokenizer.from_pretrained(model_name)
model = LongformerModel.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 输入文件
input_file = REMOVE_BOT_FILE

# 加载 JSON 文件
with open(input_file, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 统计每个diff的token长度
token_lengths = []
filtered_data = []  # 用于存储过滤后的数据

for json_line in tqdm(data, desc="Processing JSON lines", unit="line"):
    diff = json_line.get("diff")

    if diff:
        # 对diff进行token化
        inputs = tokenizer(diff, return_tensors="pt", max_length=4096, truncation=True, padding=True)
        inputs = inputs.to(device)  # 将输入数据移动到GPU
        token_length = len(inputs['input_ids'][0])  # 获取token长度

        # 仅保存长度不超过4096的diff
        if token_length <= 4096:
            filtered_data.append(json_line)

        token_lengths.append(token_length)

# 计算并打印中位数
median_length = np.median(token_lengths)
print(f"Median token length of diff: {median_length}")

# 输出一些统计信息
print(f"Total data points: {len(token_lengths)}")
print(f"Max token length: {max(token_lengths)}")
print(f"Min token length: {min(token_lengths)}")
print(f"Average token length: {sum(token_lengths) / len(token_lengths)}")

# 保存过滤后的数据到新文件
with open(FILTERED_DIFF_FILE, 'w', encoding='utf-8') as file:
    json.dump(filtered_data, file, ensure_ascii=False, indent=4)

# 将token长度分布保存到文件
with open(TOKEN_LENGTH_DIST_FILE, 'w', encoding='utf-8') as file:
    for length in token_lengths:
        file.write(f"{length}\n")

print(f"Token length distribution has been saved to {TOKEN_LENGTH_DIST_FILE}")