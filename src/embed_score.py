# src/embed_score.py
"""
Semantic scoring utilities.

Provides:
 - merge_reference_answers(questions_df, refs_df, ref_text_col='reference_answer_text')
 - add_similarity_to_df(df, ref_col='reference_list', hyp_col='transcribed_text', ...)

Behavior:
 - Uses sentence-transformers (all-MiniLM-L6-v2) for embeddings.
 - Picks best-matching reference per row and computes similarity.
 - Produces stable output columns used by the rest of the project.
"""

from typing import List
import pandas as pd
import numpy as np
import ast
import math

# Try import sentence-transformers; provide graceful fallback if unavailable
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception as e:
    SentenceTransformer = None
    cosine_similarity = None

# Module-level model cache
_MODEL = None
_DEFAULT_MODEL = "all-MiniLM-L6-v2"

def _get_model(name: str = _DEFAULT_MODEL):
    global _MODEL
    if _MODEL is None:
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers not available. Install with: pip install sentence-transformers")
        _MODEL = SentenceTransformer(name)
    return _MODEL

# -------------------------
# merge_reference_answers
# -------------------------
def merge_reference_answers(questions_df: pd.DataFrame,
                            refs_df: pd.DataFrame,
                            ref_text_col: str = "reference_answer_text") -> pd.DataFrame:
    """
    Merge questions and references into a single dataframe.

    Returns a dataframe with columns:
      question_id, question_text, reference_list (list[str]), reference_combined (str)

    - questions_df: expected to contain question id + question text (various column names accepted)
    - refs_df: expected to contain question_id and at least one reference text column
    """
    qdf = questions_df.copy() if questions_df is not None else pd.DataFrame(columns=["question_id", "question_text"])
    rdf = refs_df.copy() if refs_df is not None else pd.DataFrame(columns=["question_id", ref_text_col])

    # normalize question columns
    qdf.columns = [str(c).strip() for c in qdf.columns]
    rdf.columns = [str(c).strip() for c in rdf.columns]

    qid_col = next((c for c in qdf.columns if str(c).strip().lower() in ("question_id", "id", "qid")), None)
    qtext_col = next((c for c in qdf.columns if str(c).strip().lower() in ("question_text", "question", "text")), None)
    if qid_col is None:
        qdf["question_id"] = qdf.index.astype(str)
        qid_col = "question_id"
    if qtext_col is None:
        qdf["question_text"] = qdf[qid_col].astype(str)
        qtext_col = "question_text"
    qdf = qdf.rename(columns={qid_col: "question_id", qtext_col: "question_text"})[["question_id", "question_text"]]

    # prepare reference grouping
    if "question_id" in rdf.columns:
        # pick reference text column if the configured one doesn't exist
        if ref_text_col not in rdf.columns:
            other_cols = [c for c in rdf.columns if c != "question_id"]
            ref_text_col_eff = other_cols[0] if other_cols else None
        else:
            ref_text_col_eff = ref_text_col

        if ref_text_col_eff:
            grp = rdf.groupby("question_id")[ref_text_col_eff].apply(list).reset_index().rename(columns={ref_text_col_eff: "reference_list"})
        else:
            grp = pd.DataFrame(columns=["question_id", "reference_list"])
    else:
        grp = pd.DataFrame(columns=["question_id", "reference_list"])

    merged = pd.merge(qdf, grp, on="question_id", how="left")

    # ensure list type and clean strings
    def _ensure_list(x):
        if isinstance(x, list):
            return [str(i).strip() for i in x if str(i).strip()]
        if pd.isna(x):
            return []
        # if it's a string, check if it's a stringified list
        if isinstance(x, str):
            s = x.strip()
            if s.startswith("[") and s.endswith("]"):
                try:
                    parsed = ast.literal_eval(s)
                    if isinstance(parsed, list):
                        return [str(i).strip() for i in parsed if str(i).strip()]
                except Exception:
                    pass
            return [s] if s else []
        return [str(x)]

    merged["reference_list"] = merged["reference_list"].apply(_ensure_list)
    merged["reference_combined"] = merged["reference_list"].apply(lambda refs: " || ".join(refs) if refs else "")

    return merged

# -------------------------
# add_similarity_to_df
# -------------------------
def add_similarity_to_df(
    df: pd.DataFrame,
    ref_col: str = "reference_list",
    hyp_col: str = "transcribed_text",
    model_name: str = _DEFAULT_MODEL,
    relevance_threshold: float = 0.20,
    weight_similarity: float = 0.7,
    weight_length: float = 0.3,
    show_progress: bool = False,
) -> pd.DataFrame:
    """
    Compute semantic similarity and blended score. Returns new DataFrame with added columns:
      - best_reference (string)
      - similarity_score (0..1)
      - semantic_similarity (alias)
      - scaled_score (0..10)
      - final_score_0_10 (alias of scaled or blended score)
      - similarities (list of per-ref similarities)
      - length_ratio
      - final_score (blended)
      - feedback (text)
    """

    if df is None or df.shape[0] == 0:
        return df

    model = None
    try:
        model = _get_model(model_name)
    except Exception:
        # we continue with None and fallback to zeros
        model = None

    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]

    # Prepare lists to populate
    best_refs = []
    similarity_scores = []
    similarities_lists = []
    scaled_scores = []
    length_ratios = []
    final_scores = []
    feedbacks = []

    # Build lists for batch encoding
    hyps = []
    refs_per_row = []

    for _, row in out.iterrows():
        # get hypothesis text (with common fallbacks)
        hyp = ""
        if hyp_col in row and pd.notna(row[hyp_col]):
            hyp = str(row[hyp_col]).strip()
        else:
            for alt in ("transcribed_text", "transcript", "answer", "response"):
                if alt in row and pd.notna(row[alt]):
                    hyp = str(row[alt]).strip()
                    break
        hyps.append(hyp)

        # get refs: could be list, string, or combined string
        refs_item = row.get(ref_col, [])
        if refs_item is None:
            refs_item = []
        # if it's a string, attempt to parse as list literal or split by separator
        if isinstance(refs_item, str):
            s = refs_item.strip()
            if s.startswith("[") and s.endswith("]"):
                try:
                    parsed = ast.literal_eval(s)
                    refs = [str(r).strip() for r in parsed if str(r).strip()]
                except Exception:
                    refs = [s]
            elif "||" in s:
                refs = [p.strip() for p in s.split("||") if p.strip()]
            else:
                refs = [s] if s else []
        elif isinstance(refs_item, (list, tuple)):
            refs = [str(r).strip() for r in refs_item if str(r).strip()]
        else:
            refs = [str(refs_item).strip()]

        # ensure at least one empty ref to keep lengths aligned
        if len(refs) == 0:
            refs = [""]

        refs_per_row.append(refs)

    # Precompute embeddings in efficient batches if model is available
    # We'll compute embeddings for all unique references to reuse them
    unique_ref_map = {}
    ref_flat_list = []
    for rlist in refs_per_row:
        for r in rlist:
            if r not in unique_ref_map:
                unique_ref_map[r] = len(ref_flat_list)
                ref_flat_list.append(r)

    # encode refs
    if model is not None and len(ref_flat_list) > 0:
        try:
            ref_embs = model.encode(ref_flat_list, convert_to_numpy=True, show_progress_bar=show_progress)
        except Exception:
            ref_embs = None
    else:
        ref_embs = None

    # encode hyps
    if model is not None:
        try:
            hyp_embs = model.encode(hyps, convert_to_numpy=True, show_progress_bar=show_progress)
        except Exception:
            hyp_embs = None
    else:
        hyp_embs = None

    # For each row, compute similarity to its references, choose best
    for i, (hyp, refs) in enumerate(zip(hyps, refs_per_row)):
        # fallback values
        best_ref = ""
        best_sim = 0.0
        sims_list = [0.0 for _ in refs]
        lr = 0.0
        scaled_sim = 0.0
        final = 0.0
        fb = "No reference"

        # if model embeddings available, compute similarities
        if hyp_embs is not None and ref_embs is not None and len(refs) > 0:
            try:
                hyp_vec = hyp_embs[i].reshape(1, -1)
                # gather ref vectors
                idxs = [unique_ref_map[r] for r in refs if r in unique_ref_map]
                if len(idxs) == 0:
                    sims_list = [0.0 for _ in refs]
                else:
                    ref_vecs = ref_embs[idxs]
                    sims = cosine_similarity(ref_vecs, hyp_vec).reshape(-1).tolist()
                    # ensure same order as refs
                    sims_list = sims
                # pick best local index
                if len(sims_list) > 0:
                    best_local = int(np.argmax(sims_list))
                    best_sim = float(sims_list[best_local]) if not math.isnan(sims_list[best_local]) else 0.0
                    best_ref = refs[best_local] if best_local < len(refs) else (refs[0] if refs else "")
                else:
                    best_sim = 0.0
                    best_ref = refs[0] if refs else ""
            except Exception:
                best_sim = 0.0
                best_ref = refs[0] if refs else ""
                sims_list = [0.0 for _ in refs]
        else:
            # model not available or failure - keep zeros and choose first ref
            best_sim = 0.0
            best_ref = refs[0] if refs else ""
            sims_list = [0.0 for _ in refs]

        # length ratio feature
        def _length_ratio_fn(h, r):
            if not h or not r:
                return 0.0
            lh = len(str(h).split())
            lr_words = len(str(r).split())
            if lh == 0 or lr_words == 0:
                return 0.0
            return min(lh, lr_words) / max(lh, lr_words)

        lr = _length_ratio_fn(hyp, best_ref)

        # scaled similarity 0..10
        scaled_sim = float(best_sim * 10.0)

        # final blended score: apply relevance threshold and blend weights
        if best_sim < relevance_threshold or (hyp is None) or (str(hyp).strip() == ""):
            final = 0.0
            fb = "Answer not relevant or no transcribed text detected — scored 0."
        else:
            final_raw = (weight_similarity * scaled_sim) + (weight_length * (lr * 10.0))
            final = float(max(0.0, min(10.0, final_raw)))
            # textual feedback
            if final >= 8.0:
                fb = "Excellent answer — clear and relevant."
            elif final >= 6.0:
                fb = "Good answer — mostly relevant, minor omissions."
            elif final >= 4.0:
                fb = "Fair answer — some relevant content but misses key points."
            else:
                fb = "Poor answer — short or partially relevant."

        # append per-row outputs
        best_refs.append(best_ref)
        similarity_scores.append(round(float(best_sim), 6))
        similarities_lists.append([round(float(s), 6) for s in sims_list])
        scaled_scores.append(round(float(scaled_sim), 2))
        length_ratios.append(round(float(lr), 3))
        final_scores.append(round(float(final), 2))
        feedbacks.append(fb)

    # write back to dataframe with consistent column names expected by the rest of the project
    out["best_reference"] = best_refs
    out["similarity_score"] = similarity_scores
    out["semantic_similarity"] = out["similarity_score"]  # alias
    out["scaled_score"] = scaled_scores
    out["final_score_0_10"] = out["scaled_score"]  # alias for older code paths
    out["similarities"] = similarities_lists
    out["length_ratio"] = length_ratios
    out["final_score"] = final_scores
    out["feedback"] = feedbacks

    return out
