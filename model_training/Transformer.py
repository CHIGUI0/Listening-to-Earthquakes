import os
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import time

import torch
import torch.nn as nn


class model_transformer(nn.Module):
    def __init__(self, input_dim, seq_len, embedding_dim, num_head, num_layers, forward_dim, dropout):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embedding_dim)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, seq_len, embedding_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_head, dim_feedforward=forward_dim, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = x + self.pos_embedding

        x = self.encoder(x)

        x = x.transpose(1, 2)
        x = self.pool(x)
        x = x.squeeze(2)
        x = self.linear(x)

        return x


train_files = [
    # '/kaggle/input/dataset/Transformer/train_features_w_150000_s_150000.npz',
    # '/kaggle/input/dataset/Transformer/train_features_w_15000_s_15000.npz',
    # '/kaggle/input/dataset/Transformer/train_features_w_15000_s_7500.npz',
    '/kaggle/input/dataset/Transformer/train_features_w_1500_s_1500.npz',
    '/kaggle/input/dataset/Transformer/train_features_w_1500_s_750.npz'
]

d = {}
d_mean_std = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
patience = 5

for train_file in train_files:
    tmp = train_file.split('train_features_')[1].split('.')[0]
    w = tmp.split('_s_')[0][2:]
    s = tmp.split('_s_')[1]

    data = np.load(train_file)
    X, y = data['X'], data['y']
    # X = np.delete(X, 4, axis=2)
    print(X.shape, y.shape, train_file)

    num_features = X.shape[2]
    seq_len = X.shape[1]

    torch.autograd.set_detect_anomaly(True)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42)

    mean = np.mean(X_train)
    std = np.std(X_train)
    std = np.clip(std, a_min=1e-8, a_max=None)

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    d_mean_std[(w, s)] = (mean, std)

    train_dataset = torch.utils.data.TensorDataset(torch.tensor(
        X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True)
    val_dataset = torch.utils.data.TensorDataset(torch.tensor(
        X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=32, shuffle=False)

    param_grid = {
        'embedding_dim': [32, 64],
        'num_head': [4, 8],
        'num_layers': [2, 4],
        'forward_dim': [64, 256],
        'dropout': [0, 0.2]
    }

    d[(w, s)] = {}

    for embedding_dim in param_grid['embedding_dim']:
        for num_head in param_grid['num_head']:
            for num_layers in param_grid['num_layers']:
                for forward_dim in param_grid['forward_dim']:
                    for dropout in param_grid['dropout']:
                        start = time.time()

                        print(
                            f"embedding_dim={embedding_dim}, num_head={num_head}, num_layers={num_layers}, forward_dim={forward_dim}, dropout={dropout}: ", end="")

                        model_path = f"/kaggle/working/{w}_{s}_{embedding_dim}_{num_head}_{num_layers}_{forward_dim}_{dropout}.pth"
                        model = model_transformer(
                            num_features, seq_len, embedding_dim, num_head, num_layers, forward_dim, dropout).to(device)
                        criterion = nn.L1Loss()
                        optimizer = torch.optim.Adam(
                            model.parameters(), lr=0.001)
                        # norm = nn.LayerNorm((seq_len, num_features)).to(device)

                        best_mae = float("inf")
                        best_model = None
                        best_epoch = 0

                        for epoch in range(100):
                            model.train()
                            for batch_X, batch_y in train_dataloader:
                                optimizer.zero_grad()
                                outputs = model(batch_X.to(device))
                                loss = criterion(outputs, batch_y.to(device))
                                loss.backward()
                                optimizer.step()

                            model.eval()
                            val_mae = 0.0
                            with torch.no_grad():
                                for batch_X, batch_y in val_dataloader:
                                    outputs = model(batch_X.to(device))
                                    val_mae += criterion(outputs, batch_y.to(
                                        device)).item() * batch_X.shape[0]

                            val_mae /= X_val.shape[0]
                            # print(f"Epoch {epoch+1}, Validation MAE: {val_mae:.4f}")

                            if val_mae < best_mae:
                                best_mae = val_mae
                                best_model = model.state_dict()
                                best_epoch = epoch + 1

                            if (epoch + 1 - best_epoch >= patience) or (epoch == 99):
                                print(
                                    f'best_mae={round(best_mae, 5)}, best_epoch={best_epoch}, time spent={round(time.time()-start, 2)}s')
                                break

                        d[(w, s)][(embedding_dim, num_head,
                                   num_layers, forward_dim, dropout)] = val_mae
                        torch.save(best_model, model_path)


for k, v in d.items():
    print(k)
    v = sorted(list(v.items()), key=lambda x: x[1])[:5]
    for line in v:
        print(line)


if not os.path.exists('/kaggle/working/predictions'):
    os.mkdir('/kaggle/working/predictions')

for (w, s), v in d.items():
    test_file = f'/kaggle/input/dataset/Transformer/test_features_w_{w}_s_{s}.npz'
    data = np.load(test_file)
    X, id = data['X'], data['seg_ids']
    num_features = X.shape[2]
    seq_len = X.shape[1]

    mean, std = d_mean_std[(w, s)]
    X = (X - mean) / std

    for k, _ in sorted(list(v.items()), key=lambda x: x[1])[:5]:
        embedding_dim, num_head, num_layers, forward_dim, dropout = k

        model_path = f"/kaggle/working/{w}_{s}_{embedding_dim}_{num_head}_{num_layers}_{forward_dim}_{dropout}.pth"
        model = model_transformer(num_features, seq_len, embedding_dim,
                                  num_head, num_layers, forward_dim, dropout).to(device)
        model.load_state_dict(torch.load(model_path, weights_only=True))

        model.eval()
        X = torch.tensor(X, dtype=torch.float32, device=device)
        y = model(X).detach().cpu().numpy()

        df = pd.DataFrame()
        df['seg_id'] = id
        df['time_to_failure'] = np.squeeze(y, 1)
        df.to_csv(
            f'/kaggle/working/predictions/{w}_{s}_{embedding_dim}_{num_head}_{num_layers}_{forward_dim}_{dropout}.csv', index=False)
