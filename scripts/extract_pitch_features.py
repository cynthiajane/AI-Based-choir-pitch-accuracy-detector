import os
import csv
import numpy as np
import librosa

# ===============================
# CONFIGURATION
# ===============================
DATASET_DIR = "dataset"
OUTPUT_CSV = "features.csv"
SAMPLE_RATE = 44100

# Reference note frequencies (same as generation)
NOTE_FREQUENCIES = {
    "C3": 130.81, "D3": 146.83, "E3": 164.81, "F3": 174.61,
    "G3": 196.00, "A3": 220.00, "B3": 246.94,
    "C4": 261.63, "D4": 293.66, "E4": 329.63, "F4": 349.23,
    "G4": 392.00, "A4": 440.00, "B4": 493.88, "C5": 523.25
}

# ===============================
# HELPER FUNCTIONS
# ===============================
def frequency_to_cents(freq, ref_freq):
    """Convert frequency difference to cents"""
    return 1200 * np.log2(freq / ref_freq)


def extract_pitch_features(audio_path, expected_freq):
    """Extract pitch features from a single audio file"""
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)

    # Pitch extraction (YIN algorithm)
    f0 = librosa.yin(
        y,
        fmin=80,
        fmax=600,
        sr=sr
    )

    # Remove unvoiced frames
    f0 = f0[~np.isnan(f0)]

    if len(f0) == 0:
        return None

    mean_pitch = np.mean(f0)
    pitch_variation = np.std(f0)
    pitch_error_cents = frequency_to_cents(mean_pitch, expected_freq)

    return mean_pitch, pitch_error_cents, pitch_variation


# ===============================
# MAIN FEATURE EXTRACTION
# ===============================
def build_feature_csv():
    rows = []

    for label in os.listdir(DATASET_DIR):
        label_path = os.path.join(DATASET_DIR, label)
        if not os.path.isdir(label_path):
            continue

        for file in os.listdir(label_path):
            if not file.endswith(".wav"):
                continue

            # Extract note name from filename (e.g., C4_in_tune_12.wav)
            note = file.split("_")[0]
            expected_freq = NOTE_FREQUENCIES.get(note)

            if expected_freq is None:
                continue

            file_path = os.path.join(label_path, file)
            features = extract_pitch_features(file_path, expected_freq)

            if features is None:
                continue

            mean_pitch, pitch_error, pitch_var = features

            rows.append([
                file,
                mean_pitch,
                pitch_error,
                pitch_var,
                label
            ])

    # Write CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename",
            "mean_pitch_hz",
            "pitch_error_cents",
            "pitch_variation",
            "label"
        ])
        writer.writerows(rows)

    print("✅ Feature extraction complete!")
    print(f"📄 CSV saved as '{OUTPUT_CSV}'")


# ===============================
# RUN
# ===============================
if __name__ == "__main__":
    build_feature_csv()
