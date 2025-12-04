# src/evaluate.py
"""
Robust metrics computation for oral evaluator.

Reads:
 - outputs/evaluation_results.csv  (primary)
 - data/csv/human_labeling.csv     (optional; preferred if present)

Writes:
 - outputs/evaluation_metrics.json
 - outputs/similarity_results.png

Usage:
  python src/evaluate.py
"""
import os
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from typing import Tuple, Optional

THIS_FILE = Path(__file__).resolve()
# If this file lives in <project>/src/evaluate.py, project root is parents[1]
PROJECT_ROOT = THIS_FILE.parent.parent if THIS_FILE.parent.name == "src" else THIS_FILE.parent

EVAL_CSV_CANDIDATES = [
    PROJECT_ROOT / "outputs" / "evaluation_results.csv",
    Path.cwd() / "outputs" / "evaluation_results.csv",
    PROJECT_ROOT / "src" / "outputs" / "evaluation_results.csv"
]
HUMAN_CSV = PROJECT_ROOT / "data" / "csv" / "human_labeling.csv"
OUT_JSON = PROJECT_ROOT / "outputs" / "evaluation_metrics.json"
OUT_PNG = PROJECT_ROOT / "outputs" / "similarity_results.png"

def safe_basename(p):
    if pd.isna(p) or p is None:
        return ""
    return Path(str(p).replace("\\", "/")).name

def try_read_csv(path: Path):
    """Try utf-8 then latin1; return DataFrame or raise."""
    try:
        return pd.read_csv(path, encoding="utf-8")
    except Exception:
        return pd.read_csv(path, encoding="latin1")

def find_eval_csv() -> Optional[Path]:
    for c in EVAL_CSV_CANDIDATES:
        if c.exists():
            return c
    return None

def dedupe_eval_df(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows and then try to dedupe by audio_path+timestamp/question_id."""
    before = len(df)
    df = df.drop_duplicates()
    key_cols = [c for c in ("audio_path", "audio_url", "timestamp", "question_id") if c in df.columns]
    if key_cols:
        df = df.drop_duplicates(subset=key_cols, keep="last")
    after = len(df)
    if before != after:
        print(f"[dedupe] removed {before-after} duplicate rows from evaluation_results.csv")
    return df

def load_and_merge(
    human_csv_path: Optional[Path],
    eval_df: pd.DataFrame
) -> Tuple[pd.DataFrame, str, str]:
    """
    Attempt to produce a dataframe that has columns: human_score and pred_score.
    Return merged_df, human_col_name_used, pred_col_name_used
    """
    # Normalize column names
    eval_df = eval_df.copy()
    eval_df.columns = [c.strip() if isinstance(c, str) else c for c in eval_df.columns]

    # First: try to load human CSV and join using audio basename or question_id
    if human_csv_path and human_csv_path.exists():
        hdf = try_read_csv(human_csv_path)
        hdf.columns = [c.strip() if isinstance(c, str) else c for c in hdf.columns]

        # Try to rename plausible columns to canonical names
        if not {"audio_path","question_id","human_score"}.issubset(set(hdf.columns)):
            possible_audio = next((c for c in hdf.columns if "audio" in c.lower()), None)
            possible_qid = next((c for c in hdf.columns if "question" in c.lower() and "id" in c.lower()), None)
            possible_score = next((c for c in hdf.columns if "human" in c.lower() and "score" in c.lower()), None)
            if possible_audio: hdf = hdf.rename(columns={possible_audio: "audio_path"})
            if possible_qid: hdf = hdf.rename(columns={possible_qid: "question_id"})
            if possible_score: hdf = hdf.rename(columns={possible_score: "human_score"})

        # compute basenames
        hdf["audio_basename"] = hdf.get("audio_path","").fillna("").apply(safe_basename)
        eval_df["audio_basename"] = eval_df.get("audio_path","").fillna("").apply(safe_basename)

        # Try merges in order:
        merged = pd.DataFrame()
        if "audio_basename" in hdf.columns and "audio_basename" in eval_df.columns and "question_id" in hdf.columns and "question_id" in eval_df.columns:
            merged = pd.merge(hdf, eval_df, left_on=["audio_basename","question_id"], right_on=["audio_basename","question_id"], how="inner")
            print(f"[DEBUG] Merge on audio_basename+question_id: {merged.shape[0]} rows")
        if merged.empty and "audio_basename" in hdf.columns and "audio_basename" in eval_df.columns:
            merged = pd.merge(hdf, eval_df, left_on="audio_basename", right_on="audio_basename", how="inner")
            print(f"[DEBUG] Merge on audio_basename only: {merged.shape[0]} rows")
        if merged.empty and "question_id" in hdf.columns and "question_id" in eval_df.columns:
            merged = pd.merge(hdf, eval_df, on="question_id", how="inner")
            print(f"[DEBUG] Merge on question_id only: {merged.shape[0]} rows")

        if not merged.empty:
            print("[DEBUG] Sample merged rows:")
            print(merged[[c for c in merged.columns if 'score' in c or 'audio' in c or 'question_id' in c]].head(5))
            # ensure numeric columns names
            if "human_score" not in merged.columns:
                for c in merged.columns:
                    if "human" in str(c).lower() and "score" in str(c).lower():
                        merged = merged.rename(columns={c: "human_score"})
                        break
            pred_candidates = ["final_score_0_10","scaled_score","final_score","pred_score","semantic_similarity"]
            pred_col = next((c for c in pred_candidates if c in merged.columns), None)
            if pred_col is None:
                numeric_cols = [c for c in eval_df.columns if pd.api.types.is_numeric_dtype(eval_df[c])]
                pred_col = numeric_cols[0] if numeric_cols else None
            if pred_col is None:
                raise RuntimeError("Could not find any prediction column in evaluation_results.csv after merging with human CSV.")
            merged = merged.rename(columns={pred_col: "pred_score"})
            merged["human_score"] = pd.to_numeric(merged["human_score"], errors="coerce")
            merged["pred_score"] = pd.to_numeric(merged["pred_score"], errors="coerce")
            merged = merged.dropna(subset=["human_score","pred_score"])
            print(f"[DEBUG] After dropna: {merged.shape[0]} rows")
            return merged, "human_score", "pred_score"

    # If no human CSV or merge failed => try to find columns inside eval_df itself
    eval_candidates_human = ["human_score","human_label","human_rating","human","score"]
    eval_candidates_pred  = ["final_score_0_10","scaled_score","final_score","pred_score","semantic_similarity"]

    human_col = next((c for c in eval_candidates_human if c in eval_df.columns), None)
    pred_col  = next((c for c in eval_candidates_pred if c in eval_df.columns), None)

    # fallback: find numeric column in 0-10 range for human
    if human_col is None:
        numeric_cols = [c for c in eval_df.columns if pd.api.types.is_numeric_dtype(eval_df[c])]
        for c in numeric_cols:
            vals = pd.to_numeric(eval_df[c], errors="coerce").dropna()
            if len(vals) and vals.between(0, 10).mean() > 0.5:
                human_col = c
                break

    # fallback: pick other numeric as pred
    if pred_col is None:
        numeric_cols = [c for c in eval_df.columns if pd.api.types.is_numeric_dtype(eval_df[c])]
        for c in numeric_cols:
            if c == human_col:
                continue
            pred_col = c
            break

    if human_col is None or pred_col is None:
        raise RuntimeError(f"No human/pred columns detected. Eval columns: {list(eval_df.columns)}")

    df_eval = eval_df.copy()
    df_eval[human_col] = pd.to_numeric(df_eval[human_col], errors="coerce")
    df_eval[pred_col]  = pd.to_numeric(df_eval[pred_col], errors="coerce")
    df_eval = df_eval.dropna(subset=[human_col, pred_col])
    df_eval = df_eval.rename(columns={human_col: "human_score", pred_col: "pred_score"})
    return df_eval, human_col, pred_col

def safe_corr(a, b, method="pearson"):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2:
        return None, None
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return None, None
    try:
        if method == "pearson":
            r, p = stats.pearsonr(a, b)
        else:
            r, p = stats.spearmanr(a, b)
        if math.isnan(r):
            return None, None
        return float(r), float(p)
    except Exception:
        return None, None

def compute_and_save_metrics(eval_csv_path: Path = None):
    # find csv
    chosen = Path(eval_csv_path) if eval_csv_path else find_eval_csv()
    if not chosen or not chosen.exists():
        print("[metrics] evaluation CSV not found; looked at candidates:")
        for c in EVAL_CSV_CANDIDATES:
            print("  -", c)
        return None

    print("[metrics] using evaluation CSV:", chosen)
    eval_df = try_read_csv(chosen)
    if len(eval_df) == 0:
        print("[metrics] evaluation CSV empty.")
        return None

    # dedupe
    eval_df = dedupe_eval_df(eval_df)

    # attempt to merge with human CSV if present
    try:
        merged, used_human_col, used_pred_col = load_and_merge(HUMAN_CSV if HUMAN_CSV.exists() else None, eval_df)
    except Exception as e:
        print("[metrics] merge/load failed:", e)
        return None

    if merged is None or len(merged) == 0:
        print("[metrics] After merge no rows to evaluate.")
        return None

    y_true = pd.to_numeric(merged["human_score"], errors="coerce").astype(float).values
    y_pred = pd.to_numeric(merged["pred_score"], errors="coerce").astype(float).values

    if len(y_true) == 0:
        print("[metrics] No numeric rows available after coercion.")
        return None

    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(math.sqrt(mse))
    within_1 = float(np.mean(np.abs(y_true - y_pred) <= 1.0))
    pear_r, pear_p = safe_corr(y_true, y_pred, "pearson")
    spear_r, spear_p = safe_corr(y_true, y_pred, "spearman")
    try:
        r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else None
    except Exception:
        r2 = None

    metrics = {
        "human_col": "human_score",
        "pred_col": "pred_score",
        "n": int(len(y_true)),
        "pearson_r": pear_r,
        "pearson_p": pear_p,
        "spearman_r": spear_r,
        "spearman_p": spear_p,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "within_1": within_1
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("[metrics] Saved JSON ->", OUT_JSON)

    # scatter plot
    try:
        plt.figure(figsize=(6,6))
        plt.scatter(y_true, y_pred, alpha=0.6)
        mn = min(np.min(y_true), np.min(y_pred))
        mx = max(np.max(y_true), np.max(y_pred))
        if mn == mx:
            mn -= 1; mx += 1
        plt.plot([mn,mx],[mn,mx],"k--")
        plt.xlabel("Human score"); plt.ylabel("Predicted score")
        plt.title(f"MAE={mae:.3f}, n={len(y_true)}")
        plt.tight_layout(); plt.savefig(OUT_PNG); plt.close()
        print("[metrics] Saved plot ->", OUT_PNG)
    except Exception as e:
        print("[metrics] Failed to save plot:", e)

    return metrics

if __name__ == "__main__":
    compute_and_save_metrics()
