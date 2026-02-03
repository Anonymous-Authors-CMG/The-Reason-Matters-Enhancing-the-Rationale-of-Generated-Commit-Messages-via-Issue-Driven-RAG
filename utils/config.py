import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configuration paths
text_model_path = 'FacebookAI/roberta-large'
code_model_path = 'microsoft/codebert-base'
model_base_path = "../saved/models"
fine_tuned_encoder_path = "../model/encoder/fine-tuned-model"
commit_classifier_model_path = "../saved/models/lstm/commit-classifier.ckpt"
vector_database_path = '../saved/vector/'
commit_classifier_tokenizer = '../model/cache/hub/bert-base-uncased'

feature_path = "../saved/features"
input_path = "../data/input"
issue_path = "../data/issue/"
embedding_path = "../data/issue/issue_embedding_large/"

github_tokens = [
    'xxxx',
]

project_type_mapping = {
    'pytorch': 'dl',
    'tensorflow': 'dl',
    'keras': 'dl',
    'kafka': 'mq',
    'rocketmq': 'mq',
    'pulsar': 'mq',
}

# high-quality commit message configs
high_quality_params = {
    "top_k": 30,
    "similarity_threshold": 0.3,
    "label_level" : 3,
}

# possible issue retrieval configs
possible_issue_params = {
    "coarse_threshold": 0.5, 
    "max_candidates": 100,
    "max_result": 1
}