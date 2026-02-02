from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import os, re, json


class CodeIssueEncoder:
    def __init__(self, base_model='microsoft/codebert-base'):
        self.model = SentenceTransformer(base_model)
    
    def prepare_training_data(self, data_path, max_length=30000):
        with open(data_path, 'r') as f:
            raw_data = json.load(f)
        
        raw_data = raw_data[:max_length]
        train_examples = []
        
        for sample in raw_data:
            code_diff = sample['diff']
            positive_issue = sample['positive_issues']['text']
            
            # 为每个负样本创建一个训练样本
            for neg_issue in sample['negative_issues']:
                train_examples.append(
                    InputExample(
                        texts=[code_diff, positive_issue, neg_issue['text']],
                        label=1.0  
                    )
                )
        
        return train_examples
    
    def train(self, train_examples, output_path, epochs=3, batch_size=8):
        train_dataloader = DataLoader(
            train_examples, 
            shuffle=True, 
            batch_size=batch_size
        )
        
        train_loss = losses.MultipleNegativesRankingLoss(self.model)
        
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=int(len(train_dataloader) * 0.1),
            output_path=output_path,
            show_progress_bar=True
        )
    
    def encode(self, texts, convert_to_tensor=True):
        return self.model.encode(texts, convert_to_tensor=convert_to_tensor)

if __name__ == "__main__":
    encoder = CodeIssueEncoder()
    train_data = encoder.prepare_training_data('data/dataset.json')
    encoder.train(train_data, output_path='./fine-tuned-model', epochs=5, batch_size=18)
