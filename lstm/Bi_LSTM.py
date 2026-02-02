import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import torch.optim as optim
from torch.nn.functional import one_hot
import pytorch_lightning as pl
from pathlib import Path
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics.functional import accuracy, recall, precision, f1_score
import os, time
import json

# ==================== 配置参数 ====================
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.set_float32_matmul_precision('high')

# 超参数配置
CONFIG = {
    'batch_size': 128,
    'epochs': 60,
    'dropout': 0.2,
    'rnn_hidden': 768,
    'rnn_layer': 2,
    'class_num': 4,
    'lr': 0.001,
    'max_length': 512,
    'bert_model': 'bert-base-uncased',
    'train_data': './data/train.csv',
    'val_data': './data/dev.csv',
    'test_data': './data/test.csv',
    'checkpoint_dir': './checkpoints',
    'log_dir': './logs'
}


# ==================== 数据集定义 ====================
class CommitDataset(Dataset):
    """
    Commit消息数据集
    CSV格式: commit, label
    label含义:
        0: "Why and What"
        1: "Neither Why nor What"
        2: "No What"
        3: "No Why"
    """

    def __init__(self, csv_path):
        self.dataset = load_dataset('csv', data_files=csv_path, split='train')

    def __getitem__(self, idx):
        commit = self.dataset[idx]['message']
        label = self.dataset[idx]['label']
        return commit, label

    def __len__(self):
        return len(self.dataset)


# ==================== 批处理函数 ====================
def collate_fn(batch, tokenizer, max_length=200):
    """
    批处理函数，将commit文本转换为BERT输入格式
    """
    commits = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    # BERT编码
    encoded = tokenizer.batch_encode_plus(
        batch_text_or_text_pairs=commits,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt',
        return_length=True,
    )

    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']
    token_type_ids = encoded['token_type_ids']
    labels = torch.LongTensor(labels)

    return input_ids, attention_mask, token_type_ids, labels


# ==================== 模型定义 ====================
class BiLSTMClassifier(nn.Module):
    """
    BERT + BiLSTM 分类器
    """

    def __init__(self, dropout, hidden_dim, output_dim, num_layers=2):
        super(BiLSTMClassifier, self).__init__()

        # BERT embedding层
        self.embedding = BertModel.from_pretrained('bert-base-uncased')

        # 冻结BERT参数
        for param in self.embedding.parameters():
            param.requires_grad_(False)

        # BiLSTM层
        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 全连接层
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # BERT编码
        embedded = self.embedding(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        embedded = embedded.last_hidden_state

        # BiLSTM
        lstm_out, (h_n, c_n) = self.lstm(embedded)

        # 连接最后一层的前向和后向隐藏状态
        output = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        output = self.dropout(output)
        output = self.fc(output)

        return output


# ==================== Lightning模块 ====================
class CommitClassifierLightning(pl.LightningModule):
    """
    PyTorch Lightning模块
    """

    def __init__(self, config):
        super(CommitClassifierLightning, self).__init__()
        self.save_hyperparameters()
        self.config = config

        # 初始化模型
        self.model = BiLSTMClassifier(
            dropout=config['dropout'],
            hidden_dim=config['rnn_hidden'],
            output_dim=config['class_num'],
            num_layers=config['rnn_layer']
        )

        # 损失函数
        self.criterion = nn.CrossEntropyLoss()

        # 初始化tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(config['bert_model'])

        # 用于存储测试结果
        self.test_predictions = []
        self.test_labels = []

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.model(input_ids, attention_mask, token_type_ids)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.config['lr'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels = batch

        # Forward pass
        logits = self.model(input_ids, attention_mask, token_type_ids)
        loss = self.criterion(logits, labels)

        # 计算准确率
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, labels, task='multiclass', num_classes=self.config['class_num'])

        # 记录指标
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_acc', acc, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels = batch

        # Forward pass
        logits = self.model(input_ids, attention_mask, token_type_ids)
        loss = self.criterion(logits, labels)

        # 计算指标
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, labels, task='multiclass', num_classes=self.config['class_num'])
        f1 = f1_score(preds, labels, task='multiclass', num_classes=self.config['class_num'], average='macro')

        # 记录指标
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_f1', f1, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels = batch

        # Forward pass
        logits = self.model(input_ids, attention_mask, token_type_ids)
        loss = self.criterion(logits, labels)

        # 计算指标
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, labels, task='multiclass', num_classes=self.config['class_num'])
        f1 = f1_score(preds, labels, task='multiclass', num_classes=self.config['class_num'], average='macro')
        prec = precision(preds, labels, task='multiclass', num_classes=self.config['class_num'], average='macro')
        rec = recall(preds, labels, task='multiclass', num_classes=self.config['class_num'], average='macro')

        # 保存预测结果
        self.test_predictions.extend(preds.cpu().tolist())
        self.test_labels.extend(labels.cpu().tolist())

        # 记录指标
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True)
        self.log('test_f1', f1, on_step=False, on_epoch=True)
        self.log('test_precision', prec, on_step=False, on_epoch=True)
        self.log('test_recall', rec, on_step=False, on_epoch=True)

        return loss

    def on_test_epoch_end(self):
        """测试结束后保存预测结果"""
        results = {
            'predictions': self.test_predictions,
            'labels': self.test_labels
        }

        # 保存到JSON文件
        with open('test_predictions.json', 'w') as f:
            json.dump(results, f, indent=4)

        print(f"\n预测结果已保存到 test_predictions.json")
        print(f"总样本数: {len(self.test_predictions)}")

    def train_dataloader(self):
        dataset = CommitDataset(self.config['train_data'])
        return DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            collate_fn=lambda x: collate_fn(x, self.tokenizer, self.config['max_length']),
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

    def val_dataloader(self):
        dataset = CommitDataset(self.config['val_data'])
        return DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            collate_fn=lambda x: collate_fn(x, self.tokenizer, self.config['max_length']),
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

    def test_dataloader(self):
        dataset = CommitDataset(self.config['test_data'])
        return DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            collate_fn=lambda x: collate_fn(x, self.tokenizer, self.config['max_length']),
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )


# ==================== 训练函数 ====================
def train_model(config):
    """
    训练模型
    """
    print("=" * 50)
    print("开始训练模型...")
    print("=" * 50)

    # 创建模型
    model = CommitClassifierLightning(config)

    # 创建保存目录
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)

    # 回调函数
    checkpoint_callback = ModelCheckpoint(
        dirpath=config['checkpoint_dir'],
        filename='commit-classifier-{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True,
        verbose=True
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=5,
        mode='min',
        verbose=True
    )

    # 训练器
    trainer = Trainer(
        max_epochs=config['epochs'],
        callbacks=[checkpoint_callback, early_stop_callback],
        default_root_dir=config['log_dir'],
        accelerator='auto',
        devices='auto',
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=True
    )

    # 开始训练
    trainer.fit(model)

    print("\n训练完成!")
    print(f"最佳模型保存在: {checkpoint_callback.best_model_path}")

    return trainer, model, checkpoint_callback.best_model_path


# ==================== 测试函数 ====================
def test_model(checkpoint_path, config):
    print("=" * 50)
    print("开始测试模型...")
    print("=" * 50)
    
    # 等待checkpoint文件完全写入
    max_wait = 30  # 最多等待30秒
    wait_time = 0
    while not Path(checkpoint_path).exists() and wait_time < max_wait:
        print(f"⏳ 等待checkpoint文件... ({wait_time}s)")
        time.sleep(1)
        wait_time += 1
    
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"❌ Checkpoint文件不存在: {checkpoint_path}")
    
    print(f"✓ 找到checkpoint: {checkpoint_path}")
    
    # 加载模型
    model = CommitClassifierLightning.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        config=config
    )
    
    # 测试器 - 只在主进程运行
    trainer = Trainer(
        accelerator='auto',
        devices=1,  # 测试时只使用1个设备
        enable_progress_bar=True,
        strategy='auto'  # 自动选择策略
    )
    
    # 开始测试
    results = trainer.test(model)
    
    print("\n测试完成!")
    print("测试结果:")
    for key, value in results[0].items():
        print(f"  {key}: {value:.4f}")
    
    return results


# ==================== 主函数 ====================
def main():
    """
    主函数：完整的训练和测试流程
    """
    print("\n" + "=" * 50 + "\n")

    # 1. 训练模型
    trainer, model, best_model_path = train_model(CONFIG)

    # 2. 测试模型
    test_results = test_model(best_model_path, CONFIG)
    print(test_results)

if __name__ == '__main__':
    main()
    # best_path = '/root/workspace/cmh/model1/model/lstm/checkpoints/commit-classifier-epoch=08-val_loss=0.8524-val_acc=0.6988.ckpt'
    # test_model(best_path, CONFIG)