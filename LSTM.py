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
# 4. 构造 PyTorch Dataset 和 LSTM 模型
# =============================================================================
class EarthquakeDataset(Dataset):
    def __init__(self, features_file, is_train=True):
        """
        features_file: 包含 X 和 y（如果是训练数据）的 npz 文件路径
        is_train: True 表示训练数据，False 表示测试数据
        """
        data = np.load(features_file, allow_pickle=True)
        self.X = data["X"]  # shape: (n_samples, n_windows, feature_dim)
        self.is_train = is_train
        if is_train:
            self.y = data["y"]  # shape: (n_samples, 1)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        sample = torch.tensor(self.X[idx], dtype=torch.float32)
        if self.is_train:
            label = torch.tensor(self.y[idx], dtype=torch.float32)
            return sample, label
        else:
            return sample

class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.5):
        """
        dropout: LSTM 后和全连接层前添加 dropout，用于正则化
        """
        super(LSTMPredictor, self).__init__()
        # 如果只有一层，nn.LSTM 的 dropout 参数无效，因此在后面添加 dropout 层
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                            num_layers=num_layers, batch_first=True, 
                            dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # x: [batch, seq_len, feature_dim]
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out

# =============================================================================
# 5. 训练流程
# =============================================================================
def train_model(window_size, window_stride, train_file_path, num_epochs=50, batch_size=32, lr=0.001, hidden_size=50, num_layers=1, dropout=0.5, weight_decay=1e-4):
    # 初始化 wandb
    experiment_name = f"w_{window_size}_s_{window_stride}_hidden_{hidden_size}_layers_{num_layers}_epochs_{num_epochs}_batch_{batch_size}_lr_{lr}_dropout_{dropout}"
    wandb.init(project="earthquake-lstm-mae", 
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
    
    input_size = dataset.X.shape[2]  # 特征维度（此处为 11）
    model = LSTMPredictor(input_size=input_size, hidden_size=hidden_size, dropout=dropout, num_layers=num_layers)
    
    # 检测是否有GPU可用
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
        # 记录 wandb
        wandb.log({"epoch": epoch+1, "train_loss": train_loss, "val_loss": val_loss})
        
    model_save_path = "models/lstm_mae_" + experiment_name + ".pt"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved as {model_save_path}")
    
    artifact = wandb.Artifact(name=experiment_name, type="model")
    # 将文件添加到 Artifact
    artifact.add_file(model_save_path)
    # 将 Artifact 记录到 wandb
    wandb.log_artifact(artifact)

    wandb.finish()
    return model, dataset

# =============================================================================
# 6. 测试代码
# =============================================================================
def test_model(model, test_features_file, test_save_path):
    """
    加载处理好的测试数据，利用训练好的模型进行预测。
    输出每个测试片段的预测值及对应的 seg_id。
    """
    test_data = np.load(test_features_file, allow_pickle=True)
    X_test = test_data["X"]  # shape: (n_samples, n_windows, feature_dim)
    seg_ids = test_data["seg_ids"]
    
    # 检测是否有GPU可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        predictions = outputs.squeeze().cpu().numpy()
    # 保存预测结果为csv文件，格式为 seg_id 和预测值
    predictions_results = []
    for seg_id, pred in zip(seg_ids, predictions):
        predictions_results.append([seg_id, pred])
    predictions_df = pd.DataFrame(predictions_results, columns=["seg_id", "time_to_failure"])
    predictions_df.to_csv(test_save_path, index=False)
    return seg_ids, predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an LSTM model for earthquake prediction.")
    parser.add_argument("--train_file_path", type=str, default="train_features.npz", help="Path to the training features file.")
    parser.add_argument("--window_size", type=int, default=150, help="Size of the sliding window.")
    parser.add_argument("--window_stride", type=int, default=30, help="Stride of the sliding window.")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--num_layers", type=int, default=1, help="Number of LSTM layers.")
    parser.add_argument("--hidden_size", type=int, default=50, help="Size of the hidden layer.")
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
        model = LSTMPredictor(input_size=11, hidden_size=args.hidden_size, num_layers=args.num_layers, dropout=args.dropout)
        model_save_path = "models/lstm_mae_" + f"w_{args.window_size}_s_{args.window_stride}_hidden_{args.hidden_size}_layers_{args.num_layers}_epochs_{args.num_epochs}_batch_{args.batch_size}_lr_{args.lr}_dropout_{args.dropout}" + ".pt"
        model.load_state_dict(torch.load(model_save_path))
        model.eval()
        
        test_save_path = "datasets/results/" + f"lstm_mae_test_results_w_{args.window_size}_s_{args.window_stride}_hidden_{args.hidden_size}_layers_{args.num_layers}_epochs_{args.num_epochs}_batch_{args.batch_size}_lr_{args.lr}_dropout_{args.dropout}" + ".csv"
        test_model(model, args.test_file_path, test_save_path)