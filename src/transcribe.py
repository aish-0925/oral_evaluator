import os, subprocess, numpy as np
import whisper
from tqdm import tqdm
from src import config

_ASR_MODEL = None

def load_asr_model(model_name=None):
    global _ASR_MODEL
    if _ASR_MODEL is not None:
        return _ASR_MODEL
    model_name = model_name or config.WHISPER_MODEL
    print(f"[ASR] Loading Whisper model: {model_name}")
    _ASR_MODEL = whisper.load_model(model_name)
    return _ASR_MODEL

def convert_to_wav_if_needed(file_path, sr=16000):
    ext = file_path.lower().split('.')[-1]
    if ext == 'wav':
        return file_path
    out = file_path.rsplit('.',1)[0] + '_conv.wav'
    cmd = ['ffmpeg','-y','-i',file_path,'-ar',str(sr),'-ac','1',out]
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return out
    except Exception as e:
        print('[ASR] ffmpeg convert failed:', e)
        return file_path

def safe_avg_logprob(segments):
    vals = [seg.get('avg_logprob') for seg in segments if seg.get('avg_logprob') is not None]
    if not vals:
        return None
    return float(np.mean(vals))

def transcribe_file(model, file_path):
    if not os.path.exists(file_path):
        return "", None
    try:
        file_path = convert_to_wav_if_needed(file_path)
        res = model.transcribe(file_path, fp16=False)
        text = res.get('text','').strip()
        conf = None
        if 'segments' in res and len(res['segments'])>0:
            avg = safe_avg_logprob(res['segments'])
            if avg is not None:
                conf = float(np.exp(avg))
        return text, conf
    except Exception as e:
        print('[ASR] transcribe_file error:', e)
        return "", None

def transcribe_from_df(df, audio_col='audio_path'):
    model = load_asr_model(config.WHISPER_MODEL)
    texts, confs = [], []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        t, c = transcribe_file(model, row[audio_col])
        texts.append(t); confs.append(c)
    df['transcribed_text'] = texts
    df['asr_confidence'] = confs
    return df
