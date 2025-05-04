import os
import time
import json
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

# -*- coding: utf-8 -*-
"""
LANL Earthquake Prediction – Feature Extraction (Extended)
--------------------------------------------------------
Adds a richer set of statistical and signal features to the original 12.
"""

import os
import time
import json
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

# =============================================================================
# 1. Helper functions: additional signal measures
# =============================================================================

def hurst_exponent(ts: np.ndarray) -> float:
    """Hurst exponent via rescaled range method over lags 2..99"""
    n = len(ts)
    if n < 20:
        return 0.5
    lags = range(2, min(100, n//2))
    tau = [np.sqrt(np.std(ts[lag:] - ts[:-lag])) for lag in lags]
    # fit log-log
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return float(poly[0])


def fractal_dimension(x: np.ndarray, kmax: int = 10) -> float:
    """Estimate fractal dimension via box-counting approximation"""
    N = len(x)
    L = []
    for k in range(1, min(kmax, N)):
        Lk = []
        for m in range(k):
            idx = np.arange(m, N, k)
            y = x[idx]
            if len(y) < 2:
                continue
            Lk.append((np.sum(np.abs(np.diff(y))) * (N - 1)) / ((len(y) - 1) * k))
        if Lk:
            L.append(np.mean(Lk))
    if len(L) < 2:
        return float('nan')
    coeffs = np.polyfit(np.log(1.0/np.arange(1, len(L)+1)), np.log(L), 1)
    return float(coeffs[0])

# =============================================================================
# 2. Extended feature extraction for a single window
# =============================================================================

def extract_all_features(x: np.ndarray) -> dict:
    """Compute a broad set of statistical and signal features from 1D window."""
    # convert & basic stats
    x = x.astype(np.float64)
    _min = x.min()
    _max = x.max()
    _mean = x.mean()
    _std = x.std()
    _median = np.median(x)

    # quantiles
    qs = {q: np.percentile(x, q) for q in (1, 5, 10, 25, 75, 90, 95, 99)}
    iqr = qs[75] - qs[25]

    features = {
        # original features
        "fractal_dimension_katz": fractal_dimension(x, kmax=10),
        "q95": qs[95],
        "zcr": np.mean(x[:-1] * x[1:] < 0),
        "mean": _mean,
        "mad": np.median(np.abs(x - _median)),
        "hurst_exponent_rs": hurst_exponent(x),
        "skewness": skew(x) if _std > 1e-12 else 0.0,
        "rise_time": float(max(np.argmax(x) - np.argmin(x), 0)),
        "std": _std,
        "kurtosis": kurtosis(x) if _std > 1e-12 else -3.0,
        "crest_factor": _max / (np.sqrt(np.mean(x**2)) + 1e-12),
        "min": _min,
    }

    # extended features
    features.update({
        "max": _max,
        "range": _max - _min,
        "median": _median,
        "q01": qs[1],
        "q05": qs[5],
        "q10": qs[10],
        "q25": qs[25],
        "q75": qs[75],
        "q90": qs[90],
        "q99": qs[99],
        "iqr": iqr,
        "rms": np.sqrt(np.mean(x**2)),
        "energy": np.sum(x**2) / len(x),
    })

    # convert numpy types
    for k, v in features.items():
        if isinstance(v, (np.generic, np.ndarray)):
            features[k] = v.item()
    return features

# =============================================================================
# 3. Window processing using sliding window
# =============================================================================

ALL_FEATURES_ORDER = [
    # keep consistent order for NumPy arrays
    "fractal_dimension_katz", "q95", "zcr", "mean", "mad",
    "hurst_exponent_rs", "skewness", "rise_time", "std", "kurtosis",
    "crest_factor", "min", 
    # extended
    "max", "range", "median", "q01", "q05", "q10", "q25", "q75",
    "q90", "q95", "q99", "iqr", "rms", "energy"
]

def process_segment_to_dicts(segment: np.ndarray, window_size: int = 1500, window_stride: int = 750) -> list[dict]:
    """Split a segment into windows and extract features as list of dicts."""
    out = []
    n = len(segment)
    if n < window_size:
        return []
    for start in range(0, n - window_size + 1, window_stride):
        w = segment[start:start + window_size]
        feats = extract_all_features(w)
        out.append(feats)
    return out

# =============================================================================
# 4. 第一步：处理数据并保存所有特征到 JSON
# =============================================================================

def generate_intermediate_json(
    train_csv: str,
    test_folder: str,
    window_size: int,
    window_stride: int,
    *,
    seg_length: int = 150_000,
    output_folder: str = "/kaggle/working/intermediate_json", # Specify a folder
):
    """
    STEP 1: Processes train and test data, saves ALL extracted features 
            for a given window/stride configuration into JSON files.
    """
    print(f"\n--- Generating Intermediate JSON for W={window_size}, S={window_stride} ---")
    start_time = time.time()
    os.makedirs(output_folder, exist_ok=True) # Create output folder if it doesn't exist

    train_json_file = os.path.join(output_folder, f"intermediate_train_w_{window_size}_s_{window_stride}.json")
    test_json_file = os.path.join(output_folder, f"intermediate_test_w_{window_size}_s_{window_stride}.json")

    # --- Process Training Data ---
    print(f"Processing training data: {train_csv}")
    df_train = pd.read_csv(train_csv)
    acoustic = df_train["acoustic_data"].values # Keep as is initially
    ttf = df_train["time_to_failure"].values.astype(np.float64)

    n_segments = len(acoustic) // seg_length
    train_data_for_json = []
    
    print(f"Total training segments: {n_segments}")
    for i in range(n_segments):
        start, end = i * seg_length, (i + 1) * seg_length
        seg = acoustic[start:end]
        label = ttf[end - 1].item() # Ensure label is standard python float

        # Process segment into list of feature dicts
        # Pass seg directly, conversion happens in extract_all_features
        feature_dicts_list = process_segment_to_dicts(
            seg, window_size=window_size, window_stride=window_stride
        )
        
        # Only add if feature extraction was successful (segment long enough)
        if feature_dicts_list: 
            train_data_for_json.append({
                "segment_index": i, # Keep track of original segment order
                "features": feature_dicts_list, # List of dicts for this segment
                "y": label 
            })
            
        if (i + 1) % 50 == 0 or (i + 1) == n_segments:
             print(f"  Processed training segment {i + 1}/{n_segments}")

    print(f"Saving intermediate training features to {train_json_file}")
    with open(train_json_file, 'w') as f:
        json.dump(train_data_for_json, f, indent=2) # Use indent for readability (optional)

    # --- Process Test Data ---
    print(f"Processing test data from: {test_folder}")
    test_data_for_json = {} # Use dict: {seg_id: [list of feature dicts]}
    test_files = sorted(os.listdir(test_folder))
    print(f"Total test segments: {len(test_files)}")

    for i, file_name in enumerate(test_files):
        file_path = os.path.join(test_folder, file_name)
        df_test = pd.read_csv(file_path)
        # Check if 'acoustic_data' column exists
        if "acoustic_data" not in df_test.columns:
            print(f"Warning: 'acoustic_data' column not found in {file_name}. Skipping.")
            continue
            
        seg = df_test["acoustic_data"].values
        # Seg ID: use column if exists, otherwise filename
        seg_id = df_test["seg_id"].iloc[0] if "seg_id" in df_test.columns and not df_test["seg_id"].empty else file_name.split(".")[0]

        # Process segment into list of feature dicts
        feature_dicts_list = process_segment_to_dicts(
            seg, window_size=window_size, window_stride=window_stride
        )
        
        # Only add if feature extraction was successful
        if feature_dicts_list:
            test_data_for_json[seg_id] = feature_dicts_list

        if (i + 1) % 100 == 0 or (i + 1) == len(test_files):
             print(f"  Processed test segment {i + 1}/{len(test_files)} ({file_name})")

    print(f"Saving intermediate test features to {test_json_file}")
    with open(test_json_file, 'w') as f:
        json.dump(test_data_for_json, f, indent=2) # Use indent for readability

    end_time = time.time()
    print(f"--- JSON Generation for W={window_size}, S={window_stride} finished in {end_time - start_time:.2f} seconds ---")


# =============================================================================
# 5. 第二步：从 JSON 加载并根据特征子集生成 NPZ 文件
# =============================================================================

def generate_final_npz(
    feature_subset: list[str],
    intermediate_train_json: str,
    intermediate_test_json: str,
    output_train_npz: str,
    output_test_npz: str,
    *,
    target_dtype=np.float32 # Specify desired dtype for NPZ
):
    """
    STEP 2: Loads intermediate JSON data, selects features specified in 
            `feature_subset`, and saves final training and testing NPZ files.
    """
    print(f"\n--- Generating Final NPZ for features: {', '.join(feature_subset)} ---")
    print(f"Using Train JSON: {intermediate_train_json}")
    print(f"Using Test JSON: {intermediate_test_json}")
    start_time = time.time()

    # --- Process Training Data from JSON ---
    try:
        with open(intermediate_train_json, 'r') as f:
            train_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Training JSON file not found: {intermediate_train_json}")
        return
    except json.JSONDecodeError:
         print(f"Error: Could not decode JSON from: {intermediate_train_json}")
         return

    X_train_list, y_train_list = [], []
    for segment_data in train_data:
        segment_features = []
        # Check if segment has features (might be empty if segment was too short)
        if not segment_data.get("features"):
            continue
            
        for window_features_dict in segment_data["features"]:
            # Select features based on feature_subset, maintaining order
            selected_features = []
            for feature_name in feature_subset:
                if feature_name in window_features_dict:
                    selected_features.append(window_features_dict[feature_name])
                else:
                    # Handle missing feature (should not happen if Step 1 ran correctly)
                    print(f"Warning: Feature '{feature_name}' not found in training segment {segment_data.get('segment_index', 'N/A')}, window data. Using NaN.")
                    selected_features.append(np.nan)
            segment_features.append(selected_features)
        
        # Only add if segment_features is not empty
        if segment_features:
            X_train_list.append(segment_features) # List of lists for this segment
            y_train_list.append(segment_data["y"])

    if not X_train_list:
        print("Error: No valid training data extracted from JSON.")
        # Decide whether to proceed or stop
    else:
        # Convert to NumPy array with desired dtype and shape
        # Shape: (n_segments, n_windows, n_selected_features)
        X_train = np.array(X_train_list, dtype=target_dtype)
        # Shape: (n_segments, 1)
        y_train = np.array(y_train_list, dtype=target_dtype).reshape(-1, 1)

        print(f"Saving final training features to {output_train_npz}")
        print(f"  Train X shape: {X_train.shape}, y shape: {y_train.shape}")
        np.savez_compressed(output_train_npz, X=X_train, y=y_train) # Use compression

    # --- Process Test Data from JSON ---
    try:
        with open(intermediate_test_json, 'r') as f:
            test_data = json.load(f) # Dict: {seg_id: [list of feature dicts]}
    except FileNotFoundError:
        print(f"Error: Test JSON file not found: {intermediate_test_json}")
        return
    except json.JSONDecodeError:
         print(f"Error: Could not decode JSON from: {intermediate_test_json}")
         return

    X_test_list, seg_ids_list = [], []
    # Ensure consistent order by sorting segment IDs
    sorted_seg_ids = sorted(test_data.keys())

    for seg_id in sorted_seg_ids:
        segment_features_list = test_data[seg_id]
        segment_features = []
         # Check if segment has features (might be empty if segment was too short)
        if not segment_features_list:
            print(f"Warning: No features found for test segment {seg_id}. Skipping.")
            continue

        for window_features_dict in segment_features_list:
            selected_features = []
            for feature_name in feature_subset:
                if feature_name in window_features_dict:
                    selected_features.append(window_features_dict[feature_name])
                else:
                    print(f"Warning: Feature '{feature_name}' not found in test segment {seg_id}, window data. Using NaN.")
                    selected_features.append(np.nan)
            segment_features.append(selected_features)
            
        # Only add if segment_features is not empty
        if segment_features:
            X_test_list.append(segment_features) # List of lists for this segment
            seg_ids_list.append(seg_id)

    if not X_test_list:
         print("Error: No valid test data extracted from JSON.")
         # Decide whether to proceed or stop
    else:
        # Convert to NumPy array with desired dtype and shape
        # Shape: (n_test_segments, n_windows, n_selected_features)
        X_test = np.array(X_test_list, dtype=target_dtype)
        seg_ids_test = np.array(seg_ids_list) # Array of strings

        print(f"Saving final test features to {output_test_npz}")
        print(f"  Test X shape: {X_test.shape}, seg_ids shape: {seg_ids_test.shape}")
        np.savez_compressed(output_test_npz, X=X_test, seg_ids=seg_ids_test) # Use compression

    end_time = time.time()
    print(f"--- Final NPZ Generation finished in {end_time - start_time:.2f} seconds ---")

# =============================================================================
# 5. 第二步：从 JSON 加载并根据特征子集生成 NPZ 文件
# =============================================================================

def generate_final_npz(
    feature_subset: list[str],
    intermediate_train_json: str,
    intermediate_test_json: str,
    output_train_npz: str,
    output_test_npz: str,
    *,
    target_dtype=np.float32 # Specify desired dtype for NPZ
):
    """
    STEP 2: Loads intermediate JSON data, selects features specified in 
            `feature_subset`, and saves final training and testing NPZ files.
    """
    print(f"\n--- Generating Final NPZ for features: {', '.join(feature_subset)} ---")
    print(f"Using Train JSON: {intermediate_train_json}")
    print(f"Using Test JSON: {intermediate_test_json}")
    start_time = time.time()

    # --- Process Training Data from JSON ---
    try:
        with open(intermediate_train_json, 'r') as f:
            train_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Training JSON file not found: {intermediate_train_json}")
        return
    except json.JSONDecodeError:
         print(f"Error: Could not decode JSON from: {intermediate_train_json}")
         return

    X_train_list, y_train_list = [], []
    for segment_data in train_data:
        segment_features = []
        # Check if segment has features (might be empty if segment was too short)
        if not segment_data.get("features"):
            continue
            
        for window_features_dict in segment_data["features"]:
            # Select features based on feature_subset, maintaining order
            selected_features = []
            for feature_name in feature_subset:
                if feature_name in window_features_dict:
                    selected_features.append(window_features_dict[feature_name])
                else:
                    # Handle missing feature (should not happen if Step 1 ran correctly)
                    print(f"Warning: Feature '{feature_name}' not found in training segment {segment_data.get('segment_index', 'N/A')}, window data. Using NaN.")
                    selected_features.append(np.nan)
            segment_features.append(selected_features)
        
        # Only add if segment_features is not empty
        if segment_features:
            X_train_list.append(segment_features) # List of lists for this segment
            y_train_list.append(segment_data["y"])

    if not X_train_list:
        print("Error: No valid training data extracted from JSON.")
        # Decide whether to proceed or stop
    else:
        # Convert to NumPy array with desired dtype and shape
        # Shape: (n_segments, n_windows, n_selected_features)
        X_train = np.array(X_train_list, dtype=target_dtype)
        # Shape: (n_segments, 1)
        y_train = np.array(y_train_list, dtype=target_dtype).reshape(-1, 1)

        print(f"Saving final training features to {output_train_npz}")
        print(f"  Train X shape: {X_train.shape}, y shape: {y_train.shape}")
        np.savez_compressed(output_train_npz, X=X_train, y=y_train) # Use compression

    # --- Process Test Data from JSON ---
    try:
        with open(intermediate_test_json, 'r') as f:
            test_data = json.load(f) # Dict: {seg_id: [list of feature dicts]}
    except FileNotFoundError:
        print(f"Error: Test JSON file not found: {intermediate_test_json}")
        return
    except json.JSONDecodeError:
         print(f"Error: Could not decode JSON from: {intermediate_test_json}")
         return

    X_test_list, seg_ids_list = [], []
    # Ensure consistent order by sorting segment IDs
    sorted_seg_ids = sorted(test_data.keys())

    for seg_id in sorted_seg_ids:
        segment_features_list = test_data[seg_id]
        segment_features = []
         # Check if segment has features (might be empty if segment was too short)
        if not segment_features_list:
            print(f"Warning: No features found for test segment {seg_id}. Skipping.")
            continue

        for window_features_dict in segment_features_list:
            selected_features = []
            for feature_name in feature_subset:
                if feature_name in window_features_dict:
                    selected_features.append(window_features_dict[feature_name])
                else:
                    print(f"Warning: Feature '{feature_name}' not found in test segment {seg_id}, window data. Using NaN.")
                    selected_features.append(np.nan)
            segment_features.append(selected_features)
            
        # Only add if segment_features is not empty
        if segment_features:
            X_test_list.append(segment_features) # List of lists for this segment
            seg_ids_list.append(seg_id)

    if not X_test_list:
         print("Error: No valid test data extracted from JSON.")
         # Decide whether to proceed or stop
    else:
        # Convert to NumPy array with desired dtype and shape
        # Shape: (n_test_segments, n_windows, n_selected_features)
        X_test = np.array(X_test_list, dtype=target_dtype)
        seg_ids_test = np.array(seg_ids_list) # Array of strings

        print(f"Saving final test features to {output_test_npz}")
        print(f"  Test X shape: {X_test.shape}, seg_ids shape: {seg_ids_test.shape}")
        np.savez_compressed(output_test_npz, X=X_test, seg_ids=seg_ids_test) # Use compression

    end_time = time.time()
    print(f"--- Final NPZ Generation finished in {end_time - start_time:.2f} seconds ---")
if __name__ == "__main__":

if __name__ == "__main__":

    # --- 配置 ---
    # 数据路径 (确保这些路径是正确的!)
    TRAIN_CSV_PATH = "/kaggle/input/LANL-Earthquake-Prediction/train.csv" 
    TEST_FOLDER = "/kaggle/input/LANL-Earthquake-Prediction/test"

    # 中间文件和最终文件的输出目录
    INTERMEDIATE_JSON_DIR = "intermediate_json_output"
    FINAL_NPZ_DIR = "final_npz_output"
    os.makedirs(INTERMEDIATE_JSON_DIR, exist_ok=True)
    os.makedirs(FINAL_NPZ_DIR, exist_ok=True)

    # 窗口和步长配置 (与原脚本一致)
    window_configs = [
        {"w_size": 150_000, "s_size": 150_000},
        {"w_size": 15_000,  "s_size": 15_000},
        {"w_size": 15_000,  "s_size": 7_500},
        {"w_size": 1_500,   "s_size": 1_500},
        {"w_size": 1_500,   "s_size": 750},
    ]

    # --- 第一步：生成所有中间 JSON 文件 ---
    # 注意：这一步只需要为每个 (w_size, s_size) 运行一次
    # 如果 JSON 文件已存在，可以注释掉此部分以节省时间
    print("===== STEP 1: Generating Intermediate JSON Files =====")
    # Check if paths exist before starting
    if not os.path.exists(TRAIN_CSV_PATH):
        print(f"Error: Training CSV not found at {TRAIN_CSV_PATH}")
        exit()
    if not os.path.exists(TEST_FOLDER):
        print(f"Error: Test folder not found at {TEST_FOLDER}")
        exit()
        
    for config in window_configs:
        w = config["w_size"]
        s = config["s_size"]
        # 检查是否已生成，如果需要可以跳过
        train_json_path = os.path.join(INTERMEDIATE_JSON_DIR, f"intermediate_train_w_{w}_s_{s}.json")
        test_json_path = os.path.join(INTERMEDIATE_JSON_DIR, f"intermediate_test_w_{w}_s_{s}.json")
        if os.path.exists(train_json_path) and os.path.exists(test_json_path):
             print(f"JSON files for W={w}, S={s} already exist. Skipping generation.")
             continue # 跳过生成步骤

        generate_intermediate_json(
            train_csv=TRAIN_CSV_PATH,
            test_folder=TEST_FOLDER,
            window_size=w,
            window_stride=s,
            output_folder=INTERMEDIATE_JSON_DIR,
        )
    print("===== STEP 1: Finished =====")
    
    # --- 第二步：根据选择的特征子集生成 NPZ 文件 ---
    print("\n===== STEP 2: Generating Final NPZ Files from JSON =====")

    # 你需要选择你想用哪个 window/stride 配置生成的 JSON 文件
    target_w_list = [150000, 15000, 15000, 1500, 1500]
    target_s_list = [150000, 15000, 7500, 1500, 750]

    model_name = "LSTM"

    INTERMEDIATE_JSON_DIR="datasets/intermediate_json"
    FINAL_NPZ_DIR=f"datasets/final_npz/{model_name}_top_5"
    os.makedirs(FINAL_NPZ_DIR, exist_ok=True)


    top_10_feature_collections = {
        "LSTM": ["q01", "q05", "q10", "q95", "mad", "max", "q99", "q90", "iqr", "rms"],
        "MLP": ["range", "max", "min", "energy", "kurtosis", "q01", "q10", "q99", "rms", "q25"],
    }
    top_5_feature_collections = {
        "LSTM": ["q01", "q05", "q10", "q95", "mad"],
    }

    for i in range(5):
        target_w = target_w_list[i]
        target_s = target_s_list[i]
        intermediate_train_json_path = os.path.join(INTERMEDIATE_JSON_DIR, f"intermediate_train_w_{target_w}_s_{target_s}.json")
        intermediate_test_json_path = os.path.join(INTERMEDIATE_JSON_DIR, f"intermediate_test_w_{target_w}_s_{target_s}.json")

        model_feature_collections = top_5_feature_collections[model_name]

        # 为第一个特征子集生成 NPZ
        print(f"\nGenerating NPZ for subset 1 (W={target_w}, S={target_s})...")
        output_train_npz = os.path.join(FINAL_NPZ_DIR, f"train_features_w_{target_w}_s_{target_s}.npz")
        output_test_npz = os.path.join(FINAL_NPZ_DIR, f"test_features_w_{target_w}_s_{target_s}.npz")
        generate_final_npz(
            feature_subset=model_feature_collections,
            intermediate_train_json=intermediate_train_json_path,
            intermediate_test_json=intermediate_test_json_path,
            output_train_npz=output_train_npz,
            output_test_npz=output_test_npz,
        )
        
    print("===== STEP 2: Finished =====")
    print(f"\nIntermediate JSON files are in: {INTERMEDIATE_JSON_DIR}")
    print(f"Final NPZ files are in: {FINAL_NPZ_DIR}")