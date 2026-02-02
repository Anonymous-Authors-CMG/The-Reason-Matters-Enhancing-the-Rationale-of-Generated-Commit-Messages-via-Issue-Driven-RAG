import torch.nn as nn
import torch
from transformers import AutoTokenizer
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BertModel(nn.Module):
    def __init__(self, textEncoder, codeEncoder, text_hidden_size, code_hidden_size, num_class):
        super(BertModel, self).__init__()
        self.textEncoder = textEncoder
        self.codeEncoder = codeEncoder
        self.text_hidden_size = text_hidden_size
        self.code_hidden_size = code_hidden_size
        self.num_class = num_class

        for param in self.textEncoder.parameters():
            param.requires_grad = True
        for param in self.codeEncoder.parameters():
            param.requires_grad = True

        self.fc = nn.Linear(text_hidden_size + code_hidden_size,
                            int((text_hidden_size + code_hidden_size) / 2))
        self.fc1 = nn.Linear(int((text_hidden_size + code_hidden_size) / 2),
                             int((text_hidden_size + code_hidden_size) / 4))
        self.fc2 = nn.Linear(int((text_hidden_size + code_hidden_size) / 4), num_class)

    def forward(self, text_input_ids=None, code_input_ids=None, labels=None):
        text_output = self.textEncoder(text_input_ids, attention_mask=text_input_ids.ne(1))[1]
        code_output = self.codeEncoder(code_input_ids, attention_mask=code_input_ids.ne(1))[1]
        combine_output = torch.cat([text_output, code_output], dim=-1)
        logits = self.fc(combine_output)
        logits = self.fc1(logits)
        logits = self.fc2(logits)
        prob = torch.softmax(logits, -1)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob


text_tokenizer = None
code_tokenizer = None

# 初始化 Tokenizer
def init_tokenizers(text_model_path, code_model_path):
    global text_tokenizer, code_tokenizer
    text_tokenizer = AutoTokenizer.from_pretrained(text_model_path)
    code_tokenizer = AutoTokenizer.from_pretrained(code_model_path)

# 编码输入（依赖已经初始化的 tokenizer）
def encode_inputs(issue_text, commit_code):
    if text_tokenizer is None or code_tokenizer is None:
        raise RuntimeError("Tokenizers have not been initialized. Call init_tokenizers() first.")
    text_tokens = text_tokenizer(issue_text, padding="max_length", max_length=512, truncation=True,
                                 return_tensors="pt")
    code_tokens = code_tokenizer(commit_code, padding="max_length", max_length=512, truncation=True,
                                 return_tensors="pt")
    return text_tokens["input_ids"].to(device), code_tokens["input_ids"].to(device)


# 从 BertModel 中提取特征
def extract_feature(model, issue_text, commit_code, cpu=False):
    text_input_ids, code_input_ids = encode_inputs(issue_text, commit_code)
    with torch.no_grad():
        text_output = model.textEncoder(text_input_ids, attention_mask=text_input_ids.ne(1))[1]
        code_output = model.codeEncoder(code_input_ids, attention_mask=code_input_ids.ne(1))[1]
        feature_vector = torch.cat([text_output, code_output], dim=-1)
    return feature_vector.cpu().numpy() if cpu else feature_vector
