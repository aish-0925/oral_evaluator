# src/api.py
import os
import datetime
import pandas as pd
from typing import Optional, Dict, Any

from .config import DATA_CSV, REFERENCE_CSV
from .transcribe import load_asr_model, transcribe_file
from .embed_score import add_similarity_to_df
from .utils import ensure_dir, append_evaluation_log

# module cache
_q_df = None
_ref_df = None
_merged_df = None


def _load_data() -> pd.DataFrame:
    """
    Load questions and references, return DataFrame with columns:
      question_id, question_text, reference_list (list), reference_combined (str)
    Caches results in module-level variable.
    """
    global _q_df, _ref_df, _merged_df
    if _merged_df is not None:
        return _merged_df

    # load questions
    if os.path.exists(DATA_CSV):
        try:
            _q_df = pd.read_csv(DATA_CSV, encoding="utf-8", engine="python")
        except Exception:
            _q_df = pd.read_csv(DATA_CSV, encoding="latin1", engine="python")
    else:
        _q_df = pd.DataFrame(columns=["question_id", "question_text"])

    # canonicalize question columns
    _q_df.columns = [str(c).strip() for c in _q_df.columns]
    qid_col = next((c for c in _q_df.columns if str(c).strip().lower() in ("question_id", "id", "qid")), None)
    qtext_col = next((c for c in _q_df.columns if str(c).strip().lower() in ("question_text", "question", "text")), None)
    if qid_col is None:
        _q_df["question_id"] = _q_df.index.astype(str)
        qid_col = "question_id"
    if qtext_col is None:
        _q_df["question_text"] = _q_df[qid_col].astype(str)
        qtext_col = "question_text"
    _q_df = _q_df.rename(columns={qid_col: "question_id", qtext_col: "question_text"})[["question_id", "question_text"]]
    _q_df["question_id"] = _q_df["question_id"].astype(str).str.strip()
    _q_df["question_text"] = _q_df["question_text"].astype(str).str.strip()

    # load references (if present)
    if os.path.exists(REFERENCE_CSV):
        try:
            _ref_df = pd.read_csv(REFERENCE_CSV, encoding="utf-8", engine="python")
        except Exception:
            _ref_df = pd.read_csv(REFERENCE_CSV, encoding="latin1", engine="python")
        _ref_df.columns = [str(c).strip() for c in _ref_df.columns]

        # find reference text column heuristically
        ref_text_col = next(
            (c for c in _ref_df.columns if "reference" in str(c).lower() or "answer" in str(c).lower()),
            None
        )
        # ensure question_id present in ref df (if not, try to find a candidate)
        if "question_id" not in _ref_df.columns:
            possible_qid = next((c for c in _ref_df.columns if "question" in str(c).lower() and "id" in str(c).lower()), None)
            if possible_qid:
                _ref_df = _ref_df.rename(columns={possible_qid: "question_id"})

        if ref_text_col and "question_id" in _ref_df.columns:
            grp = _ref_df.groupby("question_id")[ref_text_col].apply(list).reset_index().rename(columns={ref_text_col: "reference_list"})
        else:
            # fallback: if reference file exists but structure unknown, produce empty group
            grp = pd.DataFrame(columns=["question_id", "reference_list"])
    else:
        grp = pd.DataFrame(columns=["question_id", "reference_list"])

    merged = pd.merge(_q_df, grp, on="question_id", how="left")
    # normalize reference_list to actual lists
    def _safe_list(x):
        if isinstance(x, list):
            return [str(i) for i in x if str(i).strip()]
        if pd.isna(x) or x is None:
            return []
        s = str(x).strip()
        return [s] if s else []
    merged["reference_list"] = merged.get("reference_list", []).apply(_safe_list)
    merged["reference_combined"] = merged["reference_list"].apply(lambda refs: " || ".join(refs) if refs else "")
    _merged_df = merged
    return _merged_df


def _make_audio_url(fs_path: str) -> str:
    fname = os.path.basename(fs_path)
    return f"/audio/{fname}"


def evaluate_audio(audio_path: str, question_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Evaluate a single audio file against reference(s) for a question.

    Returns a dict with consistent keys for the frontend. Raises exceptions on
    import-level failures so the Flask route can return a descriptive error.
    """
    df = _load_data()

    # select the row for the question or fallback
    if question_id:
        sel = df[df["question_id"] == str(question_id)]
        if sel.shape[0] == 0:
            sel = pd.DataFrame([{"question_id": str(question_id), "question_text": "", "reference_list": [], "reference_combined": ""}])
    else:
        if df.shape[0] == 0:
            sel = pd.DataFrame([{"question_id": "Q000", "question_text": "", "reference_list": [], "reference_combined": ""}])
        else:
            sel = df.head(1)

    sel = sel.copy().reset_index(drop=True)

    # Load ASR model (load_asr_model should ideally cache internally)
    try:
        model = load_asr_model()
    except TypeError:
        # fallback: some load_asr_model implementations expect a model name param
        try:
            from .config import WHISPER_MODEL
            model = load_asr_model(WHISPER_MODEL)
        except Exception:
            model = load_asr_model("base")
    except Exception as e:
        raise RuntimeError(f"Failed to load ASR model: {e}")

    # Transcribe
    try:
        transcript_text, asr_conf = transcribe_file(model, audio_path)
    except Exception as e:
        # don't abort evaluation — proceed with empty transcript but warn
        transcript_text, asr_conf = "", None
        print("[WARN] ASR transcription failed:", e)

    sel["transcribed_text"] = transcript_text

    # Compute similarity & score using embedder helper
    try:
        tmp = add_similarity_to_df(sel, ref_col="reference_combined", hyp_col="transcribed_text")
    except Exception as e:
        print("[WARN] add_similarity_to_df failed:", e)
        tmp = sel.copy()
        # ensure expected columns exist on fallback
        tmp["semantic_similarity"] = 0.0
        tmp["similarity_score"] = 0.0
        tmp["scaled_score"] = 0.0
        tmp["best_reference"] = tmp.get("reference_combined", "")

    if tmp.shape[0] == 0:
        raise RuntimeError("Similarity pipeline returned no rows for the question.")

    row0 = tmp.iloc[0]

    # Try multiple candidate column names (different versions of embed_score may return different names)
    semantic_similarity = None
    for c in ("semantic_similarity", "similarity_score"):
        if c in row0.index:
            try:
                semantic_similarity = float(row0.get(c, 0.0))
            except Exception:
                semantic_similarity = 0.0
            break
    if semantic_similarity is None:
        semantic_similarity = 0.0

    # scaled score candidates
    scaled_score = None
    for c in ("scaled_score", "final_score_0_10", "final_score", "scaled", "pred_score"):
        if c in row0.index:
            try:
                scaled_score = float(row0.get(c, 0.0))
            except Exception:
                scaled_score = 0.0
            break
    if scaled_score is None:
        scaled_score = 0.0

    best_ref = row0.get("best_reference", row0.get("reference_combined", "")) or ""

    out = {
        "question_id": str(row0.get("question_id", question_id or "")),
        "question_text": str(row0.get("question_text", "")),
        "transcript": transcript_text,
        "asr_confidence": asr_conf,
        "best_reference": best_ref,
        "semantic_similarity": float(semantic_similarity),
        "final_score_0_10": float(scaled_score),
        "audio_path": audio_path,
        "audio_url": _make_audio_url(audio_path),
        "timestamp": datetime.datetime.utcnow().isoformat()
    }

    # Append evaluation log (best-effort) but don't fail evaluation on logging error
    try:
        log_row = {
            "timestamp": out["timestamp"],
            "question_id": out["question_id"],
            "question_text": out["question_text"],
            "transcript": out["transcript"],
            "asr_confidence": out["asr_confidence"],
            "best_reference": out["best_reference"],
            "semantic_similarity": out["semantic_similarity"],
            "final_score_0_10": out["final_score_0_10"],
            "audio_path": out["audio_path"],
            "audio_url": out["audio_url"],
        }
        append_evaluation_log(log_row)
    except Exception as e:
        # log to console but don't raise
        print("[WARN] append_evaluation_log failed:", e)

    return out


def load_questions() -> list:
    """
    Return questions for UI as [{'question_id':..., 'question_text':...}, ...]
    """
    if os.path.exists(DATA_CSV):
        try:
            qdf = pd.read_csv(DATA_CSV, encoding="utf-8", engine="python")
        except Exception:
            qdf = pd.read_csv(DATA_CSV, encoding="latin1", engine="python")
    else:
        return []

    qdf.columns = [str(c).strip() for c in qdf.columns]
    qid_col = next((c for c in qdf.columns if str(c).strip().lower() in ("question_id", "id", "qid")), None)
    qtext_col = next((c for c in qdf.columns if str(c).strip().lower() in ("question_text", "question", "text")), None)
    if qid_col is None:
        qdf["question_id"] = qdf.index.astype(str)
        qid_col = "question_id"
    if qtext_col is None:
        qdf["question_text"] = qdf[qid_col].astype(str)
        qtext_col = "question_text"
    qdf = qdf.rename(columns={qid_col: "question_id", qtext_col: "question_text"})
    qdf["question_id"] = qdf["question_id"].astype(str).str.strip()
    qdf["question_text"] = qdf["question_text"].astype(str).str.strip()
    qdf = qdf[(qdf["question_id"] != "") & (qdf["question_text"] != "")]
    return qdf[["question_id", "question_text"]].to_dict(orient="records")
