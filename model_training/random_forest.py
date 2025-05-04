import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

TRAIN_FILE = "/kaggle/input/listening-to-earthquakes/train_features_w_1500_s_750.npz"
TEST_FILE = "/kaggle/input/listening-to-earthquakes/test_features_w_1500_s_750.npz"
SUBMISSION_FILE = "/kaggle/working/submission.csv"

# Load training data from .npz file
train_npz = np.load(TRAIN_FILE)
X_train_data = train_npz['X']  # Features with shape (4194, 199, 11)
y_train_data = train_npz['y']  # Target (time_to_failure) with shape (4194, 1)

# Reshape 3D X_train_data to 2D: (samples, timesteps * features)
n_samples, n_timesteps, n_features = X_train_data.shape
X_train_2d = X_train_data.reshape(n_samples, n_timesteps * n_features)

# Convert to DataFrame and Series for consistency with original code
X = pd.DataFrame(X_train_2d)
y = pd.Series(y_train_data.flatten())  # Flatten 2D (4194, 1) to 1D (4194,)

# Split data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)


# Validate model
y_pred = model.predict(X_val)
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)
print(f"Validation MAE: {mae:.5f}")
print(f"Validation R^2 Score: {r2:.5f}")

# Load test data from .npz file
test_npz = np.load(TEST_FILE)
X_test_data = test_npz['X']      # Test features
seg_ids = test_npz['seg_ids']    # Segment IDs

# Reshape test data to 2D if it’s 3D
if X_test_data.ndim == 3:
    n_test_samples, n_test_timesteps, n_test_features = X_test_data.shape
    X_test_2d = X_test_data.reshape(n_test_samples, n_test_timesteps * n_test_features)
else:
    X_test_2d = X_test_data  # Already 2D

# Process test data
submission_list = []

# Predict for each test segment
for i in tqdm(range(len(X_test_2d)), desc="Processing test segments"):
    features = X_test_2d[i].reshape(1, -1)  # Reshape to 2D array for prediction
    predicted_ttf = model.predict(features)[0]
    submission_list.append({"seg_id": seg_ids[i], "time_to_failure": predicted_ttf})

# Create submission DataFrame
submission = pd.DataFrame(submission_list)

# Save submission
submission.to_csv(SUBMISSION_FILE, index=False)
print(f"Submission file saved: {SUBMISSION_FILE}")