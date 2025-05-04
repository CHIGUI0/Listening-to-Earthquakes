import os
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
import wandb
import argparse
from tqdm import tqdm

# =============================================================================
# 1. 数据集定义（与 LSTM 版本相同）
# =============================================================================
class EarthquakeDataset(Dataset):
    def __init__(self, features_file, is_train=True, scaler=None):
        """
        features_file: 包含 X 和 y（如果是训练数据）的 npz 文件路径
        is_train: True 表示训练数据，False 表示测试数据
        scaler: 用于标准化特征的 StandardScaler 对象，若为 None，则内部拟合
        """
        data = np.load(features_file, allow_pickle=True)
        self.X = data["X"]  # shape: (n_samples, n_windows, feature_dim)
        self.is_train = is_train
        if is_train:
            self.y = data["y"]  # shape: (n_samples, 1)

        # 如有需要，可添加特征标准化的代码
        # n_samples, n_windows, feature_dim = self.X.shape
        # X_2d = self.X.reshape(n_samples * n_windows, feature_dim)
        # if scaler is None:
        #     self.scaler = StandardScaler()
        #     X_scaled = self.scaler.fit_transform(X_2d)
        # else:
        #     self.scaler = scaler
        #     X_scaled = self.scaler.transform(X_2d)
        # self.X = X_scaled.reshape(n_samples, n_windows, feature_dim)
        # if is_train:
        #     self.label_scaler = StandardScaler()
        #     self.y = self.label_scaler.fit_transform(self.y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        sample = torch.tensor(self.X[idx], dtype=torch.float32)
        if self.is_train:
            label = torch.tensor(self.y[idx], dtype=torch.float32)
            return sample, label
        else:
            return sample

# =============================================================================
# 2. 定义 MLP 模型（支持可配置层数 num_layers）
# =============================================================================
class MLPPredictor(nn.Module):
    def __init__(self, n_windows, feature_dim, hidden_size, num_layers, dropout=0.5):
        """
        n_windows: 每个样本的窗口数
        feature_dim: 每个窗口的特征维度
        hidden_size: 隐藏层大小
        num_layers: 全连接隐藏层数（至少为 1）
        dropout: dropout 比例，用于防止过拟合
        """
        super(MLPPredictor, self).__init__()
        input_size = n_windows * feature_dim  # 展平后的总特征数
        layers = []
        # 第一层：输入层 -> hidden_size
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        # 后续 num_layers-1 个隐藏层
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        # 输出层
        layers.append(nn.Linear(hidden_size, 1))
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        # x: [batch, n_windows, feature_dim]
        x = x.view(x.size(0), -1)  # 展平成 [batch, n_windows * feature_dim]
        return self.model(x)

# =============================================================================
# 3. 训练流程（MLP 版本）
# =============================================================================
def train_model(window_size, window_stride, train_file_path, num_epochs=50, batch_size=32, lr=0.001, hidden_size=50, num_layers=1, dropout=0.5, weight_decay=1e-4):
    # 初始化 wandb，并自定义实验名称
    experiment_name = f"MLP_w_{window_size}_s_{window_stride}_hidden_{hidden_size}_layers_{num_layers}_epochs_{num_epochs}_batch_{batch_size}_lr_{lr}_dropout_{dropout}"
    wandb.init(project="earthquake-mlp-mae-top10", 
               name=experiment_name,
               config={
                   "window_size": window_size,
                   "window_stride": window_stride,
                   "num_epochs": num_epochs,
                   "batch_size": batch_size,
                   "lr": lr,
                   "hidden_size": hidden_size,
                   "num_layers": num_layers,
                   "dropout": dropout,
                   "weight_decay": weight_decay
               })
    config = wandb.config

    features_file = train_file_path
    dataset = EarthquakeDataset(features_file, is_train=True)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    # 获取数据形状，确定 MLP 的输入维度
    n_samples, n_windows, feature_dim = dataset.X.shape
    model = MLPPredictor(n_windows=n_windows, feature_dim=feature_dim, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
    
    # 检测 GPU 可用性
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")
    
    # 使用 MAE 损失函数
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    for epoch in tqdm(range(num_epochs), desc="Training", unit="epoch"):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_losses.append(loss.item())
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        wandb.log({"epoch": epoch+1, "train_loss": train_loss, "val_loss": val_loss})
        
    model_save_path = "models/mlp_mae_" + experiment_name + ".pt"
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved as {model_save_path}")
    
    artifact = wandb.Artifact(name=experiment_name, type="model")
    artifact.add_file(model_save_path)
    wandb.log_artifact(artifact)
    
    wandb.finish()
    return model, dataset

# =============================================================================
# 4. 测试代码（与 LSTM 版本相同）
# =============================================================================
def test_model(model, test_save_path, X_test, seg_ids):
    """
    加载处理好的测试数据，利用训练好的模型进行预测。
    输出每个测试片段的预测值及对应的 seg_id。
    """
    # 检测 GPU 可用性
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        predictions = outputs.squeeze().cpu().numpy()
    # 保存预测结果为 csv 文件
    predictions_results = []
    for seg_id, pred in zip(seg_ids, predictions):
        predictions_results.append([seg_id, pred])
    predictions_df = pd.DataFrame(predictions_results, columns=["seg_id", "time_to_failure"])
    predictions_df.to_csv(test_save_path, index=False)
    return seg_ids, predictions

# =============================================================================
# 主程序入口
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an MLP model for earthquake prediction.")
    parser.add_argument("--train_file_path", type=str, default="train_features.npz", help="Path to the training features file.")
    parser.add_argument("--window_size", type=int, default=150, help="Size of the sliding window.")
    parser.add_argument("--window_stride", type=int, default=30, help="Stride of the sliding window.")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--hidden_size", type=int, default=50, help="Hidden layer size for MLP.")
    parser.add_argument("--num_layers", type=int, default=1, help="Number of MLP hidden layers.")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay (L2 penalty).")
    parser.add_argument("--test", action="store_true", help="Test the model on test data.")
    parser.add_argument("--test_file_path", type=str, default="test_features.npz", help="Path to the test features file.")
   
    args = parser.parse_args()
    
    if not args.test:
        train_model(
            window_size=args.window_size,
            window_stride=args.window_stride,
            train_file_path=args.train_file_path,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            weight_decay=args.weight_decay
        )
    else:
        test_data = np.load(args.test_file_path, allow_pickle=True)
        X_test = test_data["X"]  # shape: (n_samples, n_windows, feature_dim)
        seg_ids = test_data["seg_ids"]
        n_samples, n_windows, feature_dim = X_test.shape
        model = MLPPredictor(n_windows=n_windows, feature_dim=10, hidden_size=args.hidden_size, num_layers=args.num_layers, dropout=args.dropout)
        model_save_path = "models/mlp_mae_MLP_" + f"w_{args.window_size}_s_{args.window_stride}_hidden_{args.hidden_size}_layers_{args.num_layers}_epochs_{args.num_epochs}_batch_{args.batch_size}_lr_{args.lr}_dropout_{args.dropout}" + ".pt"
        model.load_state_dict(torch.load(model_save_path))
        test_save_path = "datasets/results/" + f"mlp_mae_test_results_w_{args.window_size}_s_{args.window_stride}_hidden_{args.hidden_size}_layers_{args.num_layers}_epochs_{args.num_epochs}_batch_{args.batch_size}_lr_{args.lr}_dropout_{args.dropout}" + ".csv"
        test_model(model, test_save_path=test_save_path, X_test=X_test, seg_ids=seg_ids)