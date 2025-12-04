import os
import numpy as np
import subprocess
from src.utils import load_audio_resample, save_audio
import librosa

def convert_to_wav(input_path, sr=16000):
    """
    Converts webm/mp3/m4a/etc. to WAV (16 kHz mono).
    """
    ext = input_path.lower().split(".")[-1]
    if ext == "wav":
        return input_path

    out_path = input_path.rsplit(".", 1)[0] + "_conv.wav"
    command = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ar", str(sr),
        "-ac", "1",
        out_path
    ]
    try:
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return out_path
    except Exception as e:
        print(f"[Audio] Conversion failed: {e}")
        return input_path  # fallback


def trim_silence(y, top_db=35):
    """
    Removes leading/trailing silence using librosa.
    """
    try:
        yt, _ = librosa.effects.trim(y, top_db=top_db)
        return yt
    except Exception:
        return y


def normalize_rms_safe(y, target_rms=0.08):
    """
    Normalize RMS safely with peak protection.
    """
    rms = np.sqrt(np.mean(y ** 2))
    if rms == 0:
        return y

    scale = target_rms / rms
    y_scaled = y * scale

    # Prevent clipping
    peak = np.max(np.abs(y_scaled))
    if peak > 1.0:
        y_scaled = y_scaled / peak

    return y_scaled


def process_and_save(input_path, output_path, sr=16000, target_rms=0.08):
    """
    Fully safe processing pipeline:
    - Convert → wav
    - Resample
    - Trim silence
    - Normalize RMS (safe)
    - Save
    """
    try:
        # Convert if needed
        wav_path = convert_to_wav(input_path, sr=sr)

        # Load audio
        y, _ = load_audio_resample(wav_path, sr=sr)
        if y is None or len(y) == 0:
            print(f"[Audio] Empty audio after loading: {input_path}")
            return False

        # Trim silence
        y = trim_silence(y)

        # Safe RMS normalization
        y = normalize_rms_safe(y, target_rms)

        # Save
        save_audio(output_path, y, sr)
        return True

    except Exception as e:
        print(f"[Audio] Processing failed for {input_path}: {e}")
        return False
