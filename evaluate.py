#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 * Copyright (c) 2026.
 * All rights reserved.
 * Code for SayNext project

SayNext-Bench Unified Evaluation Script
========================================
Computes the following metrics for next-utterance prediction results:

  LO-B   : BLEU-4
  LO-R   : ROUGE-L
  SS-B   : BERTScore F1  (microsoft/deberta-xlarge-mnli)
  SS-S   : Sentence-BERT Cosine Similarity  (all-mpnet-base-v2)
  CEC-V  : Continuous Emotion Consistency — Valence
  CEC-A  : Continuous Emotion Consistency — Arousal
  More mtrics TBD...

Input
-----
A directory of CSV files.  Each CSV must contain at least two columns:
  - answer     : ground-truth next utterance
  - prediction : model-generated next utterance
  (column names can be overridden with --answer_col / --pred_col)

For CEC metrics, each CSV must also contain a `video` column whose values
follow the format  <dialogue_id>/<turn_id>-<...>  (e.g. d001/t03-frame42.mp4).

Usage
-----
  # All six metrics:
  python evaluate.py --csv_dir ./results --lexicon_path ./lexicons/NRC-VADI-Lexicon.csv

  # Skip CEC (no lexicon available):
  python evaluate.py --csv_dir ./results --skip_cec

  # Full options:
  python evaluate.py --help

Dependencies
------------
  pip install sacrebleu rouge-score bert-score sentence-transformers pandas tqdm

  Models (auto-downloaded from HuggingFace on first run):
    - microsoft/deberta-xlarge-mnli  (SS-B)
    - sentence-transformers/all-mpnet-base-v2  (SS-S)

  NRC-VADI Lexicon (required for CEC-V / CEC-A):
    Download NRC-VADI-Lexicon.csv from the EmotionDynamics repository:
      https://github.com/Priya22/EmotionDynamics.git
    Place it at ./lexicons/NRC-VADI-Lexicon.csv  or pass --lexicon_path.
"""

import os
import sys
import csv
import math
import glob
import logging
import argparse
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Lexical / generation metrics ──────────────────────────────────────────────
import sacrebleu
from rouge_score import rouge_scorer
from bert_score import BERTScorer
from sentence_transformers import SentenceTransformer, util



# ══════════════════════════════════════════════════════════════════════════════
# Shared utilities
# ══════════════════════════════════════════════════════════════════════════════

def round5(x: float) -> float:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return float("nan")
    return round(x, 5)


def read_pairs_from_csv(
    path: str, answer_col: str, pred_col: str
) -> List[Tuple[str, str]]:
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if answer_col not in reader.fieldnames or pred_col not in reader.fieldnames:
            raise ValueError(
                f"Required columns '{answer_col}' and '{pred_col}' not found. "
                f"Available columns: {reader.fieldnames}"
            )
        for row in reader:
            ref = (row.get(answer_col) or "").strip()
            hyp = (row.get(pred_col) or "").strip()
            if ref == "" and hyp == "":
                continue
            pairs.append((ref, hyp))
    if not pairs:
        raise ValueError("No valid non-empty text pairs found in CSV.")
    return pairs


def maybe_lower(texts: List[str], lower: bool) -> List[str]:
    return [t.lower() for t in texts] if lower else texts


# ══════════════════════════════════════════════════════════════════════════════
# LO-B : BLEU-4
# ══════════════════════════════════════════════════════════════════════════════

def compute_bleu4(hyps: List[str], refs: List[str]) -> float:
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    return bleu.score / 100.0  # normalise to [0, 1]


# ══════════════════════════════════════════════════════════════════════════════
# LO-R : ROUGE-L
# ══════════════════════════════════════════════════════════════════════════════

def compute_rougeL(hyps: List[str], refs: List[str]) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [scorer.score(r, h)["rougeL"].fmeasure for h, r in zip(hyps, refs)]
    return sum(scores) / len(scores)


# ══════════════════════════════════════════════════════════════════════════════
# SS-B : BERTScore F1
# ══════════════════════════════════════════════════════════════════════════════

def compute_bertscore_f1(
    csv_path: str,
    answer_col: str,
    pred_col: str,
    model_type: str = "microsoft/deberta-xlarge-mnli",
    lang: str = "en",
    rescale_with_baseline: bool = False,
    out_dir: Optional[str] = None,
) -> Tuple[float, float, float]:
    """Returns (P_avg, R_avg, F1_avg)."""
    df = pd.read_csv(csv_path)
    if answer_col not in df.columns or pred_col not in df.columns:
        raise ValueError(
            f"BERTScore requires columns '{answer_col}' and '{pred_col}'. "
            f"Found: {list(df.columns)}"
        )

    scorer = BERTScorer(
        model_type=model_type,
        lang=lang,
        rescale_with_baseline=rescale_with_baseline,
    )

    P_vals, R_vals, F1_vals = [], [], []
    for _, row in df.iterrows():
        cand, ref = row.get(pred_col), row.get(answer_col)
        if not (isinstance(cand, str) and isinstance(ref, str)):
            P_vals.append(0.0); R_vals.append(0.0); F1_vals.append(0.0)
        else:
            P, R, F1 = scorer.score([cand], [ref])
            P_vals.append(float(P.mean())); R_vals.append(float(R.mean()))
            F1_vals.append(float(F1.mean()))

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        df_out = df.copy()
        df_out["BERTScore_P"] = P_vals
        df_out["BERTScore_R"] = R_vals
        df_out["BERTScore_F1"] = F1_vals
        df_out.to_csv(os.path.join(out_dir, os.path.basename(csv_path)), index=False)

    n = len(F1_vals)
    return (
        sum(P_vals) / n if n else float("nan"),
        sum(R_vals) / n if n else float("nan"),
        sum(F1_vals) / n if n else float("nan"),
    )


# ══════════════════════════════════════════════════════════════════════════════
# SS-S : Sentence-BERT Cosine Similarity
# ══════════════════════════════════════════════════════════════════════════════

_sbert_model: Optional[SentenceTransformer] = None

def _get_sbert(model_name: str) -> SentenceTransformer:
    global _sbert_model
    if _sbert_model is None:
        _sbert_model = SentenceTransformer(model_name)
    return _sbert_model


def compute_sbert_cosine(
    refs: List[str],
    hyps: List[str],
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
) -> float:
    model = _get_sbert(model_name)
    ref_embs = model.encode(refs, convert_to_tensor=True, batch_size=32, show_progress_bar=False)
    hyp_embs = model.encode(hyps, convert_to_tensor=True, batch_size=32, show_progress_bar=False)
    cos_sims = util.cos_sim(ref_embs, hyp_embs).diagonal()
    return cos_sims.mean().item()



# ══════════════════════════════════════════════════════════════════════════════
# CEC : Continuous Emotion Consistency  (Valence / Arousal)
# ══════════════════════════════════════════════════════════════════════════════

_CEC_DIMS = ["valence", "arousal", "dominance", "intensity"]


def _extract_dialogue_turn(video_str: str) -> Optional[str]:
    try:
        part = video_str.split("/")[1]
    except IndexError:
        return None
    base = os.path.splitext(part)[0]
    tokens = base.split("-")
    return tokens[0] + "-" + tokens[1] if len(tokens) >= 2 else base


def _convert_csv_for_cec(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        turn = _extract_dialogue_turn(str(row.get("video", "")))
        rows.append({"text": row["answer"],     "speaker": "answer",     "dialogueTurn": turn})
        rows.append({"text": row["prediction"], "speaker": "prediction", "dialogueTurn": turn})
    return pd.DataFrame(rows, columns=["text", "speaker", "dialogueTurn"])


def _read_lexicon(path: str, dim_names: List[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[~df["word"].isna()]
    df = df[["word"] + dim_names]
    df["word"] = df["word"].str.lower()
    return df


def _prep_dim_lexicon(df: pd.DataFrame, dim: str) -> pd.DataFrame:
    ldf = df[["word", dim]].dropna(subset=[dim]).copy()
    ldf.drop_duplicates(subset=["word"], keep="first", inplace=True)
    ldf[dim] = ldf[dim].astype(float)
    ldf.rename(columns={dim: "val"}, inplace=True)
    ldf.set_index("word", inplace=True)
    return ldf


def _get_vals(text: str, lexdf: pd.DataFrame) -> List:
    tokens = text.lower().split()
    alpha_tokens = [w for w in tokens if w.isalpha()]
    matched = [w for w in tokens if w in lexdf.index]
    vals = [lexdf.loc[w]["val"] for w in matched]

    num_tokens = len(alpha_tokens)
    num_lex_tokens = len(matched)

    if vals:
        mn, mx = min(vals), max(vals)
        norm_vals = [(v - mn) / (mx - mn) for v in vals] if mx != mn else [0.5] * len(vals)
        avg_lex_val = sum(norm_vals) / len(norm_vals)
    else:
        avg_lex_val = 0.0

    return [num_tokens, num_lex_tokens, avg_lex_val]


def _process_df_for_dim(df: pd.DataFrame, lexdf: pd.DataFrame) -> pd.DataFrame:
    rows = [
        df.iloc[i].tolist() + _get_vals(
            row["text"] if pd.notnull(row["text"]) and str(row["text"]).strip() else "abc",
            lexdf,
        )
        for i, row in df.iterrows()
    ]
    result = pd.DataFrame(rows, columns=df.columns.tolist() + ["numTokens", "numLexTokens", "avgLexVal"])
    result["lexRatio"] = result.apply(
        lambda r: r["numLexTokens"] / r["numTokens"] if r["numTokens"] != 0 else 0, axis=1
    )
    return result


def _compute_cec_score(resdf: pd.DataFrame, p: float = 0.8) -> float:
    resdf = resdf.fillna(0)
    pred_vals  = resdf["avgLexVal"].iloc[1::2].reset_index(drop=True)
    ref_vals   = resdf["avgLexVal"].iloc[0::2].reset_index(drop=True)
    diff = pred_vals - ref_vals
    scores = 1 - (abs(diff) + p * abs(diff))
    return float(scores.mean())


def compute_cec_scores(
    csv_path: str,
    lexicon: pd.DataFrame,
    dims: List[str] = ("valence", "arousal"),
) -> Dict[str, float]:
    df = pd.read_csv(csv_path)
    data = _convert_csv_for_cec(df)

    results = {}
    for dim in dims:
        lexdf = _prep_dim_lexicon(lexicon, dim)
        resdf = _process_df_for_dim(data, lexdf)
        results[dim] = _compute_cec_score(resdf)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Unified per-file evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_one_csv(
    csv_path: str,
    answer_col: str,
    pred_col: str,
    lower: bool,
    lang: str,
    bertscore_model: str,
    bertscore_out_dir: Optional[str],
    bertscore_rescale: bool,
    sbert_model: str,
    lexicon: Optional[pd.DataFrame],
    skip_cec: bool,
) -> Dict:
    pairs = read_pairs_from_csv(csv_path, answer_col, pred_col)
    refs  = maybe_lower([r for r, _ in pairs], lower)
    hyps  = maybe_lower([h for _, h in pairs], lower)

    results = {"num_samples": len(hyps)}

    # LO-B and LO-R
    results["LO-B (BLEU-4)"]  = round5(compute_bleu4(hyps, refs))
    results["LO-R (ROUGE-L)"] = round5(compute_rougeL(hyps, refs))

    # SS-B (BERTScore F1)
    _, _, f1 = compute_bertscore_f1(
        csv_path, answer_col, pred_col,
        model_type=bertscore_model,
        lang=lang,
        rescale_with_baseline=bertscore_rescale,
        out_dir=bertscore_out_dir,
    )
    results["SS-B (BERTScore-F1)"] = round5(f1)

    # SS-S (Sentence-BERT cosine)
    results["SS-S (SBERT-Cosine)"] = round5(compute_sbert_cosine(refs, hyps, sbert_model))

    # CEC-V / CEC-A
    if not skip_cec and lexicon is not None:
        try:
            cec = compute_cec_scores(csv_path, lexicon, dims=["valence", "arousal"])
            results["CEC-V (Valence)"]  = round5(cec.get("valence", float("nan")))
            results["CEC-A (Arousal)"]  = round5(cec.get("arousal",  float("nan")))
        except Exception as e:
            results["CEC-V (Valence)"]  = float("nan")
            results["CEC-A (Arousal)"]  = float("nan")
            print(f"  [WARN] CEC failed for {os.path.basename(csv_path)}: {e}", file=sys.stderr)
    else:
        results["CEC-V (Valence)"]  = "skipped"
        results["CEC-A (Arousal)"]  = "skipped"

    return results


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SayNext-Bench evaluation: LO-B, LO-R, SS-B, SS-S, CEC-V, CEC-A",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv_dir",       required=True,  help="Directory containing result CSV files.")
    p.add_argument("--answer_col",    default="answer",     help="Column name for ground-truth utterances.")
    p.add_argument("--pred_col",      default="prediction", help="Column name for model predictions.")
    p.add_argument("--lower",         action="store_true",  help="Lowercase text before evaluation.")
    p.add_argument("--lang",          default="en",         help="Language code for BERTScore.")
    p.add_argument(
        "--bertscore_model",
        default="microsoft/deberta-xlarge-mnli",
        help="HuggingFace model for SS-B (BERTScore).",
    )
    p.add_argument(
        "--sbert_model",
        default="sentence-transformers/all-mpnet-base-v2",
        help="HuggingFace model for SS-S (Sentence-BERT cosine similarity).",
    )
    p.add_argument("--bertscore_out_dir",  default=None,   help="Optional: directory to save per-sample BERTScore CSVs.")
    p.add_argument("--bertscore_rescale",  action="store_true", help="Enable BERTScore baseline rescaling.")
    p.add_argument(
        "--lexicon_path",
        default="./lexicons/NRC-VADI-Lexicon.csv",
        help=(
            "Path to NRC-VADI-Lexicon.csv for CEC metrics. "
            "Download from: https://github.com/ShawonAshraf/EmotionDynamics "
            "and place at ./lexicons/NRC-VADI-Lexicon.csv"
        ),
    )
    p.add_argument(
        "--skip_cec",
        action="store_true",
        help="Skip CEC-V and CEC-A (e.g. if lexicon is unavailable).",
    )
    p.add_argument(
        "--output_csv",
        default=None,
        help="Optional: path to save a summary CSV of all results.",
    )
    return p


def main():
    args = build_parser().parse_args()

    # ── Load lexicon for CEC (once) ─────────────────────────────────────────
    lexicon = None
    if not args.skip_cec:
        if os.path.isfile(args.lexicon_path):
            print(f"Loading NRC-VADI lexicon from: {args.lexicon_path}")
            lexicon = _read_lexicon(args.lexicon_path, _CEC_DIMS)
        else:
            print(
                f"[WARN] Lexicon not found at '{args.lexicon_path}'. CEC metrics will be skipped.\n"
                f"       To enable CEC, download NRC-VADI-Lexicon.csv from\n"
                f"       https://github.com/Priya22/EmotionDynamics.git\n"
                f"       and place it at: {args.lexicon_path}",
                file=sys.stderr,
            )
            args.skip_cec = True

    # ── Find CSV files ───────────────────────────────────────────────────────
    csv_files = sorted(glob.glob(os.path.join(args.csv_dir, "*.csv")))
    if not csv_files:
        print(f"[ERROR] No CSV files found in: {args.csv_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"\nFound {len(csv_files)} CSV file(s) in '{args.csv_dir}'\n")

    # ── Evaluate ─────────────────────────────────────────────────────────────
    all_results = []
    for csv_path in csv_files:
        fname = os.path.basename(csv_path)
        print(f"── Evaluating: {fname}")
        try:
            res = evaluate_one_csv(
                csv_path=csv_path,
                answer_col=args.answer_col,
                pred_col=args.pred_col,
                lower=args.lower,
                lang=args.lang,
                bertscore_model=args.bertscore_model,
                bertscore_out_dir=args.bertscore_out_dir,
                bertscore_rescale=args.bertscore_rescale,
                sbert_model=args.sbert_model,
                lexicon=lexicon,
                skip_cec=args.skip_cec,
            )
            for k, v in res.items():
                val_str = f"{v:.5f}" if isinstance(v, float) else str(v)
                print(f"   {k}: {val_str}")
            all_results.append({"file": fname, **res})
        except Exception as e:
            print(f"   [ERROR] {type(e).__name__}: {e}", file=sys.stderr)

    # ── Optional summary CSV ─────────────────────────────────────────────────
    if args.output_csv and all_results:
        summary_df = pd.DataFrame(all_results)
        summary_df.to_csv(args.output_csv, index=False)
        print(f"\nSummary saved to: {args.output_csv}")

    print("\nDone.")


if __name__ == "__main__":
    main()
