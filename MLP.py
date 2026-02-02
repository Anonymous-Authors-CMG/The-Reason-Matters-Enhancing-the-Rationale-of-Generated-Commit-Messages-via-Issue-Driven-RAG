import os, sys, re, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, f1_score
from sklearn.metrics import precision_recall_curve
from model.TransR import *

feature_path = "../saved/features"
input_path = "../data/input"
projects = ["kafka", "pulsar", "rocketmq"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# projects = ["pytorch", "tensorflow", "keras"]

class DualInputClassifier(nn.Module):
    def __init__(self, dropout_prob=0.2):
        super(DualInputClassifier, self).__init__()

        # 特征处理分支
        self.branch1 = nn.Sequential(
            nn.Linear(1792, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )

        self.branch2 = nn.Sequential(
            nn.Linear(542, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )

        # 融合层
        self.fc_layers = nn.Sequential(
            nn.Linear(512+128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(64, 1)
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x1, x2):
        out1 = self.branch1(x1)
        out2 = self.branch2(x2)
        # 特征融合
        combined = torch.cat((out1, out2), dim=1)
        # 分类器
        logits = self.fc_layers(combined)
        return logits

# 模型配置参数
def getargs(project_name):
    project_name = project_name.lower()
    return {
        "train_path" : os.path.join(feature_path, "train", f"{project_name}.csv"),
        "test_path": os.path.join(feature_path, "test", f"{project_name}.csv"),
        "dev_path": os.path.join(feature_path, "dev", f"{project_name}.csv"),
        "batch_size": 64,
        "lr": 5e-3,
        "weight_decay":3e-5,
        "dropout": 0.3,
        "epochs": 100,
        "save_path": f"../saved/models/mlps/{project_name}_mlp.pth"
    }

# 根据验证集预测结果动态计算最佳阈值
def find_best_threshold(preds, targets):
    precision, recall, thresholds = precision_recall_curve(targets, preds)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]

# 经转换后的数据集类
class DualFeatureDataset(Dataset):
    def __init__(self, df):
        self.feature1 = np.stack(df["fv1"].apply(
            lambda x: np.array(list(map(float, x.split(",")))).astype(np.float32)).values).reshape(-1, 1792)
        self.feature2 = np.stack(df["fv2"].apply(
            lambda x: np.array(list(map(float, x.split(",")))).astype(np.float32)).values).reshape(-1, 542)
        self.labels = df["label"].to_numpy()

        # 数据标准化
        self.feature1 = (self.feature1 - self.feature1.mean(axis=0)) / (self.feature1.std(axis=0) + 1e-8)
        self.feature2 = (self.feature2 - self.feature2.mean(axis=0)) / (self.feature2.std(axis=0) + 1e-8)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.feature1[idx]),
            torch.FloatTensor(self.feature2[idx]),
            torch.FloatTensor([self.labels[idx]])
        )


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    preds = []
    targets = []

    for x1, x2, y in dataloader:
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            outputs = model(x1, x2)
            loss = criterion(outputs, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x1.size(0)
        preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
        targets.extend(y.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(np.array(targets) >= 0.5, np.array(preds) >= 0.5)
    auc = roc_auc_score(targets, preds)
    f1 = f1_score(np.array(targets) >= 0.5, np.array(preds) >= 0.5)
    return avg_loss, acc, auc, f1

# 返回预测结果和真实标签，用于动态阈值计算
def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    preds = []
    targets = []

    with torch.no_grad():
        for x1, x2, y in dataloader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            outputs = model(x1, x2)
            loss = criterion(outputs, y)

            total_loss += loss.item() * x1.size(0)
            preds.extend(torch.sigmoid(outputs).cpu().numpy())
            targets.extend(y.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)

    # 使用 0.5 计算临时指标（仅用于日志）
    acc = accuracy_score(targets, np.array(preds) >= 0.5)
    auc = roc_auc_score(targets, preds)
    f1 = f1_score(targets, np.array(preds) >= 0.5)

    return avg_loss, acc, auc, f1, preds, targets


def train_model(project_name):
    args = getargs(project_name)

    model = DualInputClassifier(dropout_prob=args["dropout"]).to(device)
    train_df = pd.read_csv(args["train_path"])

    # 计算正负样本比例，用于 pos_weight
    neg_samples = (train_df["label"] == 0).sum()
    pos_samples = (train_df["label"] == 1).sum()
    pos_weight = torch.tensor([neg_samples / pos_samples], device=device)

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3, verbose=True)

    train_loader = DataLoader(DualFeatureDataset(train_df), batch_size=args["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    dev_loader = DataLoader(DualFeatureDataset(pd.read_csv(args["dev_path"])), batch_size=args["batch_size"] * 2, shuffle=False, num_workers=2, pin_memory=True)

    best_auc, best_f1 = 0, 0
     # 初始化最佳阈值为0.5
    best_threshold = 0.5
    for epoch in range(args["epochs"]):
        train_loss, train_acc, train_auc, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_auc, val_f1, val_preds, val_targets = validate_epoch(model, dev_loader, criterion, device)

        # 计算动态最佳阈值
        current_threshold, current_f1 = find_best_threshold(val_preds, val_targets)

        scheduler.step(val_auc)

        print(f"Epoch {epoch + 1}/{args['epochs']}")
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | AUC: {train_auc:.4f} | F1: {train_f1:.4f}")
        print(f"Dev Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | AUC: {val_auc:.4f} | F1: {val_f1:.4f} | Best Thr (F1): {current_threshold:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_f1 = current_f1
            best_threshold = current_threshold

            torch.save({
                "epoch": epoch,
                "model": model,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_auc": best_auc,
                "best_f1": best_f1,
                "best_threshold": best_threshold
            }, args["save_path"])
            print(f"Saved new best model with AUC: {best_auc:.4f} | Threshold: {best_threshold:.4f}")


def test_model(project_name, dropout_prob=0.2, need_print=True):
    args = getargs(project_name)

    model = DualInputClassifier(dropout_prob=dropout_prob).to(device)
    checkpoint = torch.load(args["save_path"], map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    # 加载训练中保存的最佳阈值
    best_threshold = checkpoint.get("best_threshold", 0.5)
    print(f"Using Best Threshold: {best_threshold:.4f}")

    test_dataset = DualFeatureDataset(pd.read_csv(args["test_path"]))
    test_loader = DataLoader(test_dataset, batch_size=args["batch_size"] * 2, shuffle=False, num_workers=2, pin_memory=True)

    preds, targets = [], []
    with torch.no_grad():
        for x1, x2, y in test_loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            outputs = model(x1, x2)
            preds.extend(torch.sigmoid(outputs).cpu().numpy())
            targets.extend(y.cpu().numpy())

    # 使用动态阈值
    preds_binary = (np.array(preds) >= best_threshold).astype(int)

    acc = accuracy_score(targets, preds_binary)
    auc = roc_auc_score(targets, preds)
    f1 = f1_score(targets, preds_binary)
    report = classification_report(targets, preds_binary)
    if need_print:
        print(f"Test Results For Project {project_name}:")
        print(f"Acc: {acc:.4f} | AUC: {auc:.4f} | F1: {f1:.4f}")
        print(report)
    return acc, auc, f1, report, preds, targets

if __name__ == "__main__":
    for project in projects:
        print(f"Training model for project: {project}")
        train_model(project)
        print(f"Testing model for project: {project}")
        test_model(project, dropout_prob=getargs(project)["dropout"])