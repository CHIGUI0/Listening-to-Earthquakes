import os
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler

# =============================================================================
# 1. 特征提取函数
# =============================================================================
def extract_features(x):
    """Compute statistical features from seismic signal"""
    x = x.astype(np.float64)  # Ensure numeric type
    features = {
        "mean": np.mean(x),
        "std": np.std(x),
        "min": np.min(x),
        "max": np.max(x),
        "kurtosis": kurtosis(x),
        "skew": skew(x),
        "median": np.median(x),
        "q01": np.quantile(x, 0.01),
        "q05": np.quantile(x, 0.05),
        "q95": np.quantile(x, 0.95),
        "q99": np.quantile(x, 0.99),
    }
    return features

# =============================================================================
# 2. 修改后的 process_segment：支持重叠窗口处理
# =============================================================================
def process_segment(segment, window_size=1000, window_stride=1000):
    """
    将一个长度为 seg_length 的片段，
    按照 window_size 划分为若干个窗口，并按 window_stride 滑动（可重叠），
    对每个窗口提取统计特征，返回 shape (n_windows, feature_dim) 的特征矩阵。
    """
    features_list = []
    # 使用滑动窗口，从 0 到 len(segment) - window_size，步长为 window_stride
    for start in range(0, len(segment) - window_size + 1, window_stride):
        end = start + window_size
        window = segment[start:end]
        feats = extract_features(window)
        # 保证特征顺序一致
        features_list.append([
            feats["mean"], feats["std"], feats["min"], feats["max"],
            feats["kurtosis"], feats["skew"], feats["median"],
            feats["q01"], feats["q05"], feats["q95"], feats["q99"]
        ])
    return np.vstack(features_list)  # (n_windows, 11)

# =============================================================================
# 3. 处理训练数据，并保存提取的特征
# =============================================================================
def process_train_data(train_csv, seg_length=150000, window_size=1000, window_stride=1000, output_file="train_features.npz"):
    """
    处理 train.csv 文件：
      - 将连续数据划分为多个长度为 seg_length 的片段；
      - 对每个片段内部按 window_size 和 window_stride 划分窗口，提取特征序列；
      - 标签为该片段最后一个时间点的 time_to_failure。
    保存提取的特征与标签到 npz 文件。
    """
    df = pd.read_csv(train_csv)
    acoustic = df["acoustic_data"].values.astype(np.int16)
    ttf = df["time_to_failure"].values.astype(np.float64)

    n_segments = len(acoustic) // seg_length
    X_list = []
    y_list = []
    for i in range(n_segments):
        start = i * seg_length
        end = start + seg_length
        seg = acoustic[start:end]
        label = ttf[end - 1]  # 片段最后一个时间点的 time_to_failure
        feat_seq = process_segment(seg, window_size=window_size, window_stride=window_stride)
        X_list.append(feat_seq)  # 每个 feat_seq 的 shape 为 (n_windows, feature_dim)
        y_list.append(label)
        if (i+1) % 10 == 0:
            print(f"Processed {i+1}/{n_segments} segments")
    X_array = np.array(X_list)  # shape: (n_segments, n_windows, feature_dim)
    y_array = np.array(y_list).reshape(-1, 1)
    np.savez(output_file, X=X_array, y=y_array)
    print(f"Training features saved to {output_file}")

# =============================================================================
# 4. 处理测试数据，并保存提取的特征
# =============================================================================
def process_test_data(test_folder, window_size=1000, window_stride=1000, output_file="test_features.npz"):
    """
    处理 test 文件夹中的所有测试数据：
      - 每个文件对应一个长度为 150000 的片段，提取窗口统计特征；
      - 保存所有片段的 seg_id 和对应的特征序列。
    """
    seg_ids = []
    X_list = []
    file_list = sorted(os.listdir(test_folder))
    for i, file_name in enumerate(file_list):
        file_path = os.path.join(test_folder, file_name)
        df = pd.read_csv(file_path)
        # 每个测试文件中包含 'acoustic_data' 列，seg_id 可直接从文件中取或由文件名获得
        seg = df["acoustic_data"].values.astype(np.int16)
        feat_seq = process_segment(seg, window_size=window_size, window_stride=window_stride)
        if "seg_id" in df.columns:
            seg_id = df["seg_id"].iloc[0]
        else:
            seg_id = file_name.split(".")[0]
        seg_ids.append(seg_id)
        X_list.append(feat_seq)
        if (i+1) % 100 == 0:
            print(f"Processed {i+1}/{len(file_list)} test segments")
    X_array = np.array(X_list)  # (n_test, n_windows, feature_dim)
    seg_ids = np.array(seg_ids)
    np.savez(output_file, X=X_array, seg_ids=seg_ids)
    print(f"Test features saved to {output_file}")

if __name__ == "__main__":
    window_sizes = [150000, 15000, 15000, 1500, 1500]
    stride_sizes = [150000, 15000, 7500, 1500, 750]
    for i in range(len(window_sizes)):
        window_size = window_sizes[i]
        stride_size = stride_sizes[i]
        train_save_path = f"/kaggle/working/train_features_w_{window_size}_s_{stride_size}.npz"
        test_save_path = f"/kaggle/working/test_features_w_{window_size}_s_{stride_size}.npz"
        train_csv_path = "/kaggle/input/LANL-Earthquake-Prediction/train.csv"  # 替换为实际路径
        process_train_data(train_csv=train_csv_path, seg_length=150000, window_size=window_size, window_stride=stride_size, output_file=train_save_path)
        test_folder = "/kaggle/input/LANL-Earthquake-Prediction/test"  # 替换为测试数据文件夹路径
        process_test_data(test_folder=test_folder, window_size=window_size, window_stride=stride_size, output_file=test_save_path)