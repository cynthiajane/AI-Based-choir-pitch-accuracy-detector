import numpy as np
import os
import soundfile as sf

# ===============================
# CONFIGURATION
# ===============================
SAMPLE_RATE = 44100        # Audio sampling rate (Hz)
DURATION = 1.5             # Duration of each note (seconds)
OUTPUT_DIR = "dataset"     # Dataset folder

# Choir vocal pitch range (C3 to C5)
PITCHES = {
    "C3": 130.81,
    "D3": 146.83,
    "E3": 164.81,
    "F3": 174.61,
    "G3": 196.00,
    "A3": 220.00,
    "B3": 246.94,
    "C4": 261.63,   # Middle C
    "D4": 293.66,
    "E4": 329.63,
    "F4": 349.23,
    "G4": 392.00,
    "A4": 440.00,
    "B4": 493.88,
    "C5": 523.25
}

# Pitch deviation ranges (in cents)
LABELS = {
    "in_tune": (-5, 5),
    "slightly_off": (10, 30),
    "off_pitch": (50, 80),
    "perfect": (0,0)
}

SAMPLES_PER_NOTE = 20

# ===============================
# HELPER FUNCTIONS
# ===============================
def cents_to_frequency(base_freq, cents):
    """Convert cents deviation to frequency"""
    return base_freq * (2 ** (cents / 1200))


def generate_vocal_like_tone(freq, duration, sr):
    """Generate a vocal-like synthetic tone"""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Vibrato (human-like modulation)
   # vibrato_rate = 6      # Hz
   # vibrato_depth = 0.005
   # vibrato = vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t)

    # Base tone with vibrato
    signal = np.sin(2 * np.pi * freq * (t)) #t+vibrato if we were to include vibrato 

    # Add light noise (room / mic simulation)
    noise = 0.001 * np.random.randn(len(signal))
    signal += noise

    # Normalize
    signal /= np.max(np.abs(signal))
    return signal


# ===============================
# DATASET GENERATION
# ===============================
def generate_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for label in LABELS.keys():
        os.makedirs(os.path.join(OUTPUT_DIR, label), exist_ok=True)

    sample_id = 0

    for note, base_freq in PITCHES.items():
        for label, (min_cents, max_cents) in LABELS.items():
            for _ in range(SAMPLES_PER_NOTE):
                cents_shift = np.random.uniform(min_cents, max_cents)
                shifted_freq = cents_to_frequency(base_freq, cents_shift)

                audio = generate_vocal_like_tone(
                    shifted_freq, DURATION, SAMPLE_RATE
                )

                filename = f"{note}_{label}_{sample_id}.wav"
                filepath = os.path.join(OUTPUT_DIR, label, filename)

                sf.write(filepath, audio, SAMPLE_RATE)
                sample_id += 1

    print("✅ Synthetic pitch dataset generated successfully!")
    print(f"📁 Saved inside '{OUTPUT_DIR}/' folder")


# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    generate_dataset()
