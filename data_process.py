# -*- coding: utf-8 -*-
"""
LANL Earthquake Prediction – Feature Extraction (前 12 个特征)
-------------------------------------------------------------
本脚本基于用户原始代码重构，按交叉验证 MDI 重要度前 12 名重新提取特征：
    1. fractal_dimension (Katz)
    2. q95
    3. zcr (zero‑crossing rate)
    4. mean
    5. mad (median absolute deviation)
    6. hurst_exponent (R/S 估计)
    7. skewness
    8. rise_time (10 %→90 % 峰值跨度，单位: 样本数)
    9. std
   10. kurtosis
   11. crest_factor
   12. min

函数接口、I/O 流程与原脚本保持兼容；仅扩展了特征计算逻辑和维度注释。
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

# =============================================================================
# 1. 单一特征计算函数
# =============================================================================

def fractal_dimension_katz(x: np.ndarray) -> float:
    """Katz Fractal Dimension – 快速近似"""
    n = len(x)
    if n < 2:
        return 0.0
    L = np.sum(np.abs(np.diff(x)))  # trajectory length
    d = np.max(np.abs(x - x[0])) + 1e-12  # avoid log(0)
    return np.log10(n) / (np.log10(d / L) + np.log10(n))


def zero_crossing_rate(x: np.ndarray) -> float:
    """Normalized zero‑crossing rate"""
    return np.mean(x[:-1] * x[1:] < 0)


def mad(x: np.ndarray) -> float:
    """Median absolute deviation"""
    return np.median(np.abs(x - np.median(x)))


def hurst_exponent_rs(x: np.ndarray, max_lag: int = 20) -> float:
    """Hurst exponent via rescaled range (R/S) method"""
    lags = np.arange(2, min(max_lag, len(x) // 2))
    if len(lags) == 0:
        return 0.5  # default for short signals
    tau = [np.std(x[lag:] - x[:-lag]) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return 2.0 * poly[0]  # slope*2 ≈ H


def rise_time(x: np.ndarray) -> float:
    """Samples between 10 % and 90 % of peak amplitude"""
    x_min, x_max = np.min(x), np.max(x)
    span = x_max - x_min
    if span == 0:
        return 0.0
    low_th = x_min + 0.1 * span
    high_th = x_min + 0.9 * span
    idx_low = np.argmax(x > low_th)
    idx_high = np.argmax(x > high_th)
    return max(idx_high - idx_low, 0)


def crest_factor(x: np.ndarray) -> float:
    """Peak/RMS"""
    rms = np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12
    return np.max(np.abs(x)) / rms

# =============================================================================
# 2. 组合特征抽取器
# =============================================================================

def extract_features(x: np.ndarray) -> dict:
    """Compute the 12‑feature set from a 1‑D seismic window"""
    x = x.astype(np.float64)
    return {
        "fractal_dimension": fractal_dimension_katz(x),
        "q95": np.quantile(x, 0.95),
        "zcr": zero_crossing_rate(x),
        "mean": np.mean(x),
        "mad": mad(x),
        "hurst_exponent": hurst_exponent_rs(x),
        "skewness": skew(x),
        "rise_time": rise_time(x),
        "std": np.std(x),
        "kurtosis": kurtosis(x),
        "crest_factor": crest_factor(x),
        "min": np.min(x),
    }

FEATURE_ORDER = [
    "fractal_dimension",
    "q95",
    "zcr",
    "mean",
    "mad",
    "hurst_exponent",
    "skewness",
    "rise_time",
    "std",
    "kurtosis",
    "crest_factor",
    "min",
]

# =============================================================================
# 3. 修改后的 process_segment：支持重叠窗口处理
# =============================================================================

def process_segment(segment: np.ndarray, *, window_size: int = 1000, window_stride: int = 1000) -> np.ndarray:
    """Split a segment into sliding windows and extract 12‑D features (n_windows, 12)"""
    features_list = []
    for start in range(0, len(segment) - window_size + 1, window_stride):
        end = start + window_size
        window = segment[start:end]
        feats = extract_features(window)
        features_list.append([feats[k] for k in FEATURE_ORDER])
    return np.vstack(features_list)

# =============================================================================
# 4. 处理训练数据，并保存提取的特征
# =============================================================================

def process_train_data(
    train_csv: str,
    *,
    seg_length: int = 150_000,
    window_size: int = 1000,
    window_stride: int = 1000,
    output_file: str = "train_features.npz",
):
    """Segment train.csv, extract (n_windows, 12) feature sequences, save to .npz"""
    df = pd.read_csv(train_csv)
    acoustic = df["acoustic_data"].values.astype(np.int16)
    ttf = df["time_to_failure"].values.astype(np.float64)

    n_segments = len(acoustic) // seg_length
    X_list, y_list = [], []
    for i in range(n_segments):
        start, end = i * seg_length, (i + 1) * seg_length
        seg = acoustic[start:end]
        label = ttf[end - 1]
        feat_seq = process_segment(seg, window_size=window_size, window_stride=window_stride)
        X_list.append(feat_seq)
        y_list.append(label)
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{n_segments} segments")

    np.savez(
        output_file,
        X=np.array(X_list, dtype=np.float32),  # (n_segments, n_windows, 12)
        y=np.array(y_list, dtype=np.float32).reshape(-1, 1),
    )
    print(f"Training features saved to {output_file}")

# =============================================================================
# 5. 处理测试数据，并保存提取的特征
# =============================================================================

def process_test_data(
    test_folder: str,
    *,
    window_size: int = 1000,
    window_stride: int = 1000,
    output_file: str = "test_features.npz",
):
    """Extract features for every test segment and save (.npz)"""
    seg_ids, X_list = [], []
    for i, file_name in enumerate(sorted(os.listdir(test_folder))):
        file_path = os.path.join(test_folder, file_name)
        df = pd.read_csv(file_path)
        seg = df["acoustic_data"].values.astype(np.int16)
        feat_seq = process_segment(seg, window_size=window_size, window_stride=window_stride)
        seg_id = df["seg_id"].iloc[0] if "seg_id" in df.columns else file_name.split(".")[0]
        seg_ids.append(seg_id)
        X_list.append(feat_seq)
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} test segments")

    np.savez(
        output_file,
        X=np.array(X_list, dtype=np.float32),  # (n_test, n_windows, 12)
        seg_ids=np.array(seg_ids),
    )
    print(f"Test features saved to {output_file}")

# =============================================================================
# 6. CLI 批处理 – 保持与原脚本一致
# =============================================================================

if __name__ == "__main__":
    window_sizes = [150_000, 15_000, 15_000, 1_500, 1_500]
    stride_sizes = [150_000, 15_000, 7_500, 1_500, 750]

    train_csv_path = "/kaggle/input/LANL-Earthquake-Prediction/train.csv"  # 修改为实际路径
    test_folder = "/kaggle/input/LANL-Earthquake-Prediction/test"  # 修改为测试数据文件夹路径

    for w_size, s_size in zip(window_sizes, stride_sizes):
        train_save = f"/kaggle/working/train_features_w_{w_size}_s_{s_size}.npz"
        test_save = f"/kaggle/working/test_features_w_{w_size}_s_{s_size}.npz"

        process_train_data(
            train_csv=train_csv_path,
            seg_length=150_000,
            window_size=w_size,
            window_stride=s_size,
            output_file=train_save,
        )
        process_test_data(
            test_folder=test_folder,
            window_size=w_size,
            window_stride=s_size,
            output_file=test_save,
        )
