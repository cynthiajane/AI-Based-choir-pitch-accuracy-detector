import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ===============================
# CONFIGURATION
# ===============================
CSV_PATH = "features.csv"
FEATURE_COLUMNS = [
    "mean_pitch_hz",
    "pitch_error_cents",
    "pitch_variation"
]
LABEL_COLUMN = "label"

# ===============================
# LOAD DATA
# ===============================
print("📄 Loading dataset...")
data = pd.read_csv(CSV_PATH)

X = data[FEATURE_COLUMNS]
y = data[LABEL_COLUMN]

# ===============================
# TRAIN / TEST SPLIT
# ===============================
print("🔀 Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===============================
# FEATURE SCALING
# ===============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===============================
# MODEL TRAINING
# ===============================
print("🤖 Training model...")
model = LogisticRegression(
    max_iter=1000,
    multi_class="auto"
)
model.fit(X_train_scaled, y_train)

# ===============================
# EVALUATION
# ===============================
print("📊 Evaluating model...")
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {accuracy * 100:.2f}%\n")

print("📄 Classification Report:")
print(classification_report(y_test, y_pred))

print("🧩 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
