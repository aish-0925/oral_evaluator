import os
import pandas as pd
from src import config
from src.embed_score import add_similarity_to_df
from src.utils import ensure_dir

# Try to import evaluate (may fail if dependencies missing)
try:
    from src.evaluate import evaluate
except Exception:
    evaluate = None

def choose_ref_column(df):
    # Prefer multi-ref column if exists, otherwise common names
    candidates = [
        "reference_answers", "reference_answer_text", "ref1", "ref", "answer", "reference"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    # try to detect columns like ref1, ref2
    ref_cols = [c for c in df.columns if c.lower().startswith("ref")]
    if ref_cols:
        # join them into a single temporary column
        df["__joined_refs__"] = df[ref_cols].fillna("").astype(str).apply(lambda row: " || ".join([r for r in row if r.strip()]), axis=1)
        return "__joined_refs__"
    return None

def choose_hyp_column(df):
    # Hypothesis (student text) candidates
    candidates = ["transcribed_text", "transcript", "student_answer", "answer_text", "answer", "response"]
    for c in candidates:
        if c in df.columns:
            return c
    return None

def compute_scaled_score(df):
    # If scaled_score already present use it. Otherwise attempt to compute from available features.
    if "scaled_score" in df.columns:
        return df
    # Prefer 'semantic' column produced by add_similarity_to_df
    if "semantic_similarity" in df.columns:
        df["scaled_score"] = df["semantic_similarity"].astype(float) * 10.0
    elif "similarity" in df.columns:
        df["scaled_score"] = df["similarity"].astype(float) * 10.0
    else:
        # last resort: try to compute using other numeric subscores if present
        numeric_cols = [c for c in df.columns if c in ["semantic","concept_cov","nli","asr_confidence","audio_factor"]]
        if numeric_cols:
            df["scaled_score"] = df[numeric_cols].mean(axis=1) * 10.0
        else:
            # fallback: set NaN
            df["scaled_score"] = pd.NA
    return df

def run_pipeline():
    data_csv = getattr(config, "DATA_CSV", None)
    if not data_csv or not os.path.exists(data_csv):
        print(f"[RUN] config.DATA_CSV not set or file missing: {data_csv}")
        return

    print("[RUN] Loading dataset:", data_csv)
    df = pd.read_csv(data_csv, encoding="latin1")
    # keep original columns as-is but create a lowercase map for flexible lookup
    cols_original = list(df.columns)
    cols_lower = {c.lower(): c for c in cols_original}

    print("[INFO] Available columns:", cols_original)

    # detect ref and hyp columns smartly
    ref_col = choose_ref_column(df)
    hyp_col = choose_hyp_column(df)

    if ref_col is None:
        print("❌ No suitable reference column found (tried reference_answers, ref1, answer, etc.).")
        print("Please provide a 'reference_answers' or 'answer' column.")
        return

    if hyp_col is None:
        print("⚠ No hypothesis column (transcribed_text) found. We'll compute similarity using the same text column (self-similarity).")
        hyp_col = ref_col

    print(f"[INFO] Using reference column: '{ref_col}' and hypothesis column: '{hyp_col}'")

    # run similarity
    try:
        df_similarity = add_similarity_to_df(df.copy(), ref_col=ref_col, hyp_col=hyp_col)
    except Exception as e:
        print("[ERROR] add_similarity_to_df failed:", e)
        return

    # ensure output dir exists
    out_path = getattr(config, "OUTPUT_CSV", "outputs/similarity_results.csv")
    ensure_dir(out_path)

    # compute scaled_score if missing
    df_similarity = compute_scaled_score(df_similarity)

    # Save results
    df_similarity.to_csv(out_path, index=False)
    print(f"[RUN] Results saved to {out_path}")
    print("✅ Text-only similarity computation complete.")

    # --- Evaluation attempt ---
    pred_col = "scaled_score" if "scaled_score" in df_similarity.columns else None

    # Candidate human columns to check (in order)
    human_candidates = [
        getattr(config, "COLUMN_HUMAN_SCORE", None),
        "human_score", "human_rating", "score", "rating", "human"
    ]
    human_col = None
    for c in human_candidates:
        if c and c in df_similarity.columns:
            human_col = c
            break

    if evaluate is not None and human_col and pred_col:
        print(f"[EVAL] Found human column '{human_col}' and prediction column '{pred_col}'. Running evaluation...")
        try:
            results = evaluate(df_similarity, pred_col=pred_col, human_col=human_col)
            print("[EVAL] Evaluation results:")
            for k,v in results.items():
                print(f"  {k}: {v}")
        except Exception as e:
            print("[EVAL] evaluate() failed:", e)
    else:
        if evaluate is None:
            print("[EVAL] evaluate() not available (import failed).")
        if human_col is None:
            print("[EVAL] No human score column found in dataset. Can't compute Pearson/MAE/etc.")
        if pred_col is None:
            print("[EVAL] No prediction column 'scaled_score' found. Here's a quick summary of available columns:")
            print(df_similarity.columns.tolist())

if __name__ == "__main__":
    run_pipeline()
