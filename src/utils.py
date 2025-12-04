# src/utils.py
import os
from pathlib import Path
import json
import soundfile as sf
import numpy as np
import csv
from typing import Dict

# try to import librosa if available
try:
    import librosa
except Exception:
    librosa = None

# src/utils.py (partial) - replace append_evaluation_log and ensure_dir

# src/utils.py

def ensure_dir(path):
    d = os.path.dirname(path) if os.path.splitext(path)[1] else path
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def append_evaluation_log(result_dict: Dict, csv_path: str = 'outputs/evaluation_results.csv'):
    """
    Append a flattened, consistently-ordered row to evaluation_results.csv.
    - Uses a fixed header so subsequent parsing is stable.
    - Uses csv.QUOTE_ALL to prevent transcript commas/newlines from breaking rows.
    """
    ensure_dir(csv_path)
    # canonical header - keep the most useful fields in a stable order
    header = [
        'timestamp',
        'question_id',
        'question_text',
        'transcript',
        'asr_confidence',
        'best_reference',
        'semantic_similarity',
        'final_score_0_10',
        'audio_path',
        'audio_url'
    ]

    # ensure all keys exist in the row (avoid KeyError)
    row = {k: result_dict.get(k, "") for k in header}

    write_header = not os.path.exists(csv_path)
    # append mode + newline=''
    with open(csv_path, 'a', newline='', encoding='utf-8') as cf:
        writer = csv.DictWriter(cf, fieldnames=header, quoting=csv.QUOTE_ALL)
        if write_header:
            writer.writeheader()
        writer.writerow(row)



# --------- question loader ---------
DEFAULT_PATHS = [
    "data/csv/os_questions.csv",
    "data/os_questions.csv",
    "os_questions.csv"
]
POSSIBLE_ID_COLS = ["question_id", "questionid", "id", "qid"]
POSSIBLE_TEXT_COLS = ["question_text", "questiontext", "question", "text", "Question"]

def _try_read_csv(path):
    import pandas as pd
    encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252"]
    seps = [",", ";", "\t"]
    last_err = None
    for enc in encodings:
        for sep in seps:
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep)
                return df
            except Exception as e:
                last_err = e
    # final fallback
    try:
        df = pd.read_csv(path)
        return df
    except Exception:
        raise last_err

def load_questions(path=None):
    import pandas as pd
    p = path or next((x for x in DEFAULT_PATHS if Path(x).exists()), None)
    if p is None:
        print(f"[load_questions] No questions file found. Tried: {DEFAULT_PATHS}")
        return []
    try:
        df = _try_read_csv(p)
    except Exception as e:
        print(f"[load_questions] Failed to read {p}: {e}")
        return []
    if df is None or df.shape[0] == 0:
        print(f"[load_questions] File {p} loaded but empty.")
        return []

    df.columns = [str(c).strip() for c in df.columns]

    id_col = None
    for cand in POSSIBLE_ID_COLS:
        for col in df.columns:
            if str(col).strip().lower() == cand:
                id_col = col
                break
        if id_col:
            break

    text_col = None
    for cand in POSSIBLE_TEXT_COLS:
        for col in df.columns:
            if str(col).strip().lower() == cand.lower():
                text_col = col
                break
        if text_col:
            break

    if id_col is None or text_col is None:
        print(f"[load_questions] Required columns missing. Found columns: {list(df.columns)}")
        return []

    df = df[[id_col, text_col]].copy()
    df[id_col] = df[id_col].astype(str).str.strip()
    df[text_col] = df[text_col].astype(str).str.strip()
    df = df[df[id_col] != ""]
    df = df[df[text_col] != ""]

    if df.shape[0] == 0:
        print(f"[load_questions] File {p} has no usable rows after cleaning.")
        return []

    # canonical keys
    out = df.rename(columns={id_col: 'question_id', text_col: 'question_text'})[['question_id', 'question_text']]
    records = out.to_dict(orient='records')
    print(f"[load_questions] Loaded {len(records)} questions from {p}")
    return records

# --------- audio helpers ----------
def load_audio_resample(path, sr=16000):
    if librosa is None:
        try:
            y, sr_read = sf.read(path)
            if y.ndim > 1:
                y = np.mean(y, axis=1)
            if sr_read != sr and librosa is not None:
                y = librosa.resample(y.astype(np.float32), orig_sr=sr_read, target_sr=sr)
                return y.astype(np.float32), sr
            return y.astype(np.float32), sr_read
        except Exception as e:
            print('load_audio_resample fallback failed:', e)
            return np.array([], dtype=np.float32), sr
    try:
        y, sr_loaded = librosa.load(path, sr=sr, mono=True)
        return y.astype(np.float32), sr_loaded
    except Exception as e:
        print('load_audio_resample failed:', e)
        return np.array([], dtype=np.float32), sr

def save_audio(path, y, sr=16000):
    ensure_dir(path)
    try:
        sf.write(path, y, sr)
    except Exception as e:
        print('save_audio failed:', e)
