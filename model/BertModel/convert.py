import torch
from transformers import AutoTokenizer
import pandas as pd
import os

projects = ["tensorflow", "pytorch", "keras", "pulsar", "kafka", "rocketmq"]
path_dict = {}
for item in projects:
    path_dict[item] = f"{item}_checkpoint-best"

feature_path = "../../saved/features"
input_path = "../../data/input"
text_model_path = 'roberta-large'
code_model_path = 'microsoft/codebert-base'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载 Tokenizer
text_tokenizer = AutoTokenizer.from_pretrained(text_model_path)
code_tokenizer = AutoTokenizer.from_pretrained(code_model_path)

# 编码输入
def encode_inputs(issue_text, commit_code):
    text_tokens = text_tokenizer(issue_text, padding="max_length", max_length=16384, truncation=True, return_tensors="pt")
    code_tokens = code_tokenizer(commit_code, padding="max_length", max_length=16384, truncation=True, return_tensors="pt")
    return text_tokens["input_ids"].to(device), code_tokens["input_ids"].to(device)

# 提取特征
def extract_feature(model, issue_text, commit_code, cpu=False):
    text_input_ids, code_input_ids = encode_inputs(issue_text, commit_code)
    with torch.no_grad():
        text_output = model.module.textEncoder(text_input_ids, attention_mask=text_input_ids.ne(1))[1]
        code_output = model.module.codeEncoder(code_input_ids, attention_mask=code_input_ids.ne(1))[1]
        feature_vector = torch.cat([text_output, code_output], dim=-1)
    return feature_vector.cpu().numpy() if cpu else feature_vector

def convert_data(project_name, suffix="_TRAIN"):
    # 先加载对应的 BTLink 模型
    model_path = os.path.join("../../saved/models", path_dict[project_name.lower()], "model.bin")
    model = torch.load(model_path, map_location=device, weights_only=False).to(device)
    model.eval()
    print(f"loaded {project_name.lower()} model from {model_path}...")
    data = pd.read_csv(os.path.join(input_path, f"{project_name + suffix}.csv"),
                       usecols=["label", "Issue_KEY", "Issue_Text", "Commit_Code"])
    data["Issue_Text"] = data["Issue_Text"].fillna("")
    data["Commit_Code"] = data["Commit_Code"].fillna("")

    df = pd.read_csv(os.path.join(feature_path, suffix[1:].lower(), f"{project_name.lower()}.csv"))
    for i, row in data.iterrows():
        fv1 = extract_feature(model, row["Issue_Text"], row["Commit_Code"], cpu=True)
        fv1_str = ",".join([str(x) for x in fv1[0]])
        df.loc[i]['fv1'] = fv1_str
    df.to_csv(os.path.join(feature_path, suffix[1:].lower(), f"{project_name.lower()}.csv"), index=False)
    print(f"Already convert features for project {project_name}...")

if __name__ == '__main__':
    # suffixs = ["_TRAIN", "_DEV", "_TEST"]
    # for suffix in suffixs:
    #     for item in projects:
    #         convert_data(item, suffix=suffix)
    convert_data("tensorflow", suffix="_TEST")