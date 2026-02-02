from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
import os, sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 参数
transr_embedding_dim = 256
transr_num_epochs = 100

base_dir = os.path.abspath("../data/input/tsv/")
train_set = os.path.join(base_dir, "train_data.tsv")
test_set = os.path.join(base_dir, "test_data.tsv")
model_dir = os.path.join(base_dir, "transR_model.pkl")

training_factory = TriplesFactory.from_path(train_set)
testing_factory = TriplesFactory.from_path(test_set)

def train_transR_model():
    result = pipeline(
        model='TransR',
        optimizer='adam',
        model_kwargs=dict(embedding_dim=transr_embedding_dim),
        training=training_factory,
        testing=testing_factory,
        device='cuda',
        training_kwargs=dict(num_epochs=transr_num_epochs),
    )
    model = result.model
    torch.save(model, model_dir)

def get_triple_embeddings(triple, flat=True, model=None):
    if model is None:
        model = load_transR_model()
    try:
        head_id = training_factory.entity_to_id[triple[0]]
        rel_id = training_factory.relation_to_id[triple[1]]
        tail_id = training_factory.entity_to_id[triple[2]]
        device = model.device
        head_id_tensor = torch.tensor([head_id], device=device)
        rel_id_tensor = torch.tensor([rel_id], device=device)
        tail_id_tensor = torch.tensor([tail_id], device=device)
    
        # 使用张量索引获取 embedding
        head_emb = model.entity_representations[0](indices=head_id_tensor).cpu().detach().numpy()
        rel_emb = model.relation_representations[0](indices=rel_id_tensor).cpu().detach().numpy()
        tail_emb = model.entity_representations[0](indices=tail_id_tensor).cpu().detach().numpy()

        if not flat:
            return head_emb, rel_emb, tail_emb
        return np.concatenate([head_emb, rel_emb, tail_emb], axis=-1).flatten()

    except Exception as e:
        return np.zeros(transr_embedding_dim * 2 + 30,) if flat else (np.zeros((1, transr_embedding_dim)), np.zeros((1, 30)), np.zeros((1, transr_embedding_dim)))

def get_triple_embeddings_batch(triples, model=None):
    if model is None:
        model = load_transR_model()
    embeddings = [torch.tensor(get_triple_embeddings(triple, flat=True, model=model)) for triple in triples]
    dim = 2 * transr_embedding_dim + 30
    # 删除全零向量
    embeddings = [emb for emb in embeddings if not torch.all(emb == 0)]
    return torch.stack(embeddings) if len(embeddings) != 0 else torch.zeros((1, dim))

def pooling_embeddings(embeddings, method='mean'):
    if method == 'mean':
        return torch.mean(embeddings, dim=0)
    elif method == 'max':
        return torch.max(embeddings, dim=0).values
    elif method == 'attention':
        attention_pooling = AttentionPooling(embeddings.size(1))
        pooled_embedding, _ = attention_pooling(embeddings)
        return pooled_embedding
    else:
        raise ValueError("Unsupported pooling method: {}".format(method))

def load_transR_model():
    model = torch.load(model_dir, weights_only=False)
    return model

class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attention_weights = nn.Linear(input_dim, 1, bias=False)

    def forward(self, embeddings):
        logits = self.attention_weights(embeddings) 
        weights = torch.softmax(logits, dim=0) 
        weighted_sum = torch.sum(weights * embeddings, dim=0)
        return weighted_sum, weights

class TransRUtils:
    def __init__(self):
        self.model = load_transR_model()
        self.embedding_dim = 2 * transr_embedding_dim + 30

    @staticmethod
    def train_model():
        train_transR_model()

    @staticmethod
    def load_model():
        return load_transR_model()

    # 给定一个类，获取这个类的图嵌入向量
    def get_class_embedding(self, project_name, class_name, pooling_method='mean'):
        df = pd.read_csv(os.path.join(base_dir, f'{project_name}_triplets.tsv'), sep='\t', header=None)
        related_triples = df[(df[0] == class_name)]
        fvs = get_triple_embeddings_batch(related_triples.to_numpy().tolist(), model=self.model)
        return pooling_embeddings(fvs, method=pooling_method)

    @staticmethod
    def polling_embeddings(embeddings, method='mean', cpu=False):
        res = pooling_embeddings(embeddings, method)
        return res.cpu().detach().numpy() if cpu else res

if __name__ == "__main__":
    transr_utils = TransRUtils()
    res = transr_utils.get_class_embedding("kafka", "UniformHeterogeneousAssignmentBuilderTest")
    print(res.cpu().detach().numpy())