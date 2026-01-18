import streamlit as st
import numpy as np
import pandas as pd
import librosa

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# =========================
# MUSICAL REFERENCE UTILS
# =========================

A4_FREQ = 440.0

def hz_to_midi(f):
    return 69 + 12 * np.log2(f / A4_FREQ)

def midi_to_hz(m):
    return A4_FREQ * (2 ** ((m - 69) / 12))

def cents_error(f0, ref_freq):
    return 1200 * np.log2(f0 / ref_freq)


# =========================
# APP CONFIG
# =========================
st.set_page_config(
    page_title="Choir Pitch Accuracy Detector",
    layout="centered"
)

st.title("🎶 Choir Pitch Accuracy Detector")
st.write(
    "Upload a vocal audio file and the model will classify it as "
    "**in-tune**, **slightly off**, or **off-pitch**."
)

# =========================
# LOAD DATA & TRAIN MODEL
# =========================
FEATURE_COLUMNS = [
    "mean_pitch_hz",
    "pitch_error_cents",
    "pitch_variation"
]

@st.cache_resource
def load_model():
    data = pd.read_csv("features.csv")

    X = data[FEATURE_COLUMNS]
    y = data["label"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)

    return model, scaler

model, scaler = load_model()

#extract features 
def extract_features(audio_file, sr=44100):
    y, sr = librosa.load(audio_file, sr=sr)

    f0 = librosa.yin(
        y,
        fmin=80,
        fmax=600,
        sr=sr
    )

    f0 = f0[~np.isnan(f0)]

    if len(f0) < 10:
        return None

    # Mean pitch
    mean_pitch = np.mean(f0)

    # Pitch stability
    pitch_variation = np.std(f0)

    # Nearest musical note
    mean_midi = hz_to_midi(mean_pitch)
    nearest_midi = np.round(mean_midi)
    reference_freq = midi_to_hz(nearest_midi)

    # TRUE pitch error in cents
    pitch_error = np.mean(np.abs(cents_error(f0, reference_freq)))

    return [[mean_pitch, pitch_error, pitch_variation]]

# =========================
# FEATURE EXTRACTION
# =========================
#def extract_features(audio_file, sr=44100):
  #  y, sr = librosa.load(audio_file, sr=sr)

    #f0 = librosa.yin(
        #y,
        #fmin=80,
        #fmax=600,
        #sr=sr
    #)

    #f0 = f0[~np.isnan(f0)]

    #if len(f0) == 0:
      #  return None

    #mean_pitch = np.mean(f0)
   # pitch_variation = np.std(f0)
   # pitch_error = 0.0  # simplified reference

   # return [[mean_pitch, pitch_error, pitch_variation]]

# =========================
# FILE UPLOAD
# =========================
uploaded_file = st.file_uploader(
    "Upload a WAV file",
    type=["wav"]
)

if uploaded_file is not None:
    st.audio(uploaded_file)

    with st.spinner("Analyzing pitch..."):
        features = extract_features(uploaded_file)

        if features is None:
            st.error("Could not detect pitch in this audio.")
        else:
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]
            confidence = model.predict_proba(features_scaled)[0]

            st.success(f"🎯 **Prediction:** {prediction}")

            st.subheader("Confidence")
            for label, prob in zip(model.classes_, confidence):
                st.write(f"{label}: {prob:.2f}")
