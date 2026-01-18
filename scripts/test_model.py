import numpy as np
import pandas as pd # type: ignore
import librosa

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# ===============================
# CONFIGURATION
# ===============================
AUDIO_FILE = "dataset/in_tune/A3_in_tune_400.wav"  # CHANGE THIS
SAMPLE_RATE = 44100

FEATURE_COLUMNS = [
    "mean_pitch_hz",
    "pitch_error_cents",
    "pitch_variation"
]

# ===============================
# LOAD TRAINING DATA (to rebuild model)
# ===============================
data = pd.read_csv("features.csv")

X = data[FEATURE_COLUMNS]
y = data["label"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model (same as training script)
model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y)

# ===============================
# FEATURE EXTRACTION
# ===============================
def extract_features(audio_path):
    y_audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE)

    f0 = librosa.yin(
        y_audio,
        fmin=80,
        fmax=600,
        sr=sr
    )

    f0 = f0[~np.isnan(f0)]

    if len(f0) == 0:
        raise ValueError("No pitch detected in audio")

    mean_pitch = np.mean(f0)
    pitch_variation = np.std(f0)

    # Estimate pitch error relative to mean (simplified)
    pitch_error = 0.0

    return [[mean_pitch, pitch_error, pitch_variation]]

# ===============================
# RUN TEST
# ===============================
features = extract_features(AUDIO_FILE)
features_scaled = scaler.transform(features)

prediction = model.predict(features_scaled)
confidence = model.predict_proba(features_scaled)

print("🎵 Audio file:", AUDIO_FILE)
print("✅ Predicted class:", prediction[0])
print("📊 Confidence:", dict(zip(model.classes_, confidence[0])))
