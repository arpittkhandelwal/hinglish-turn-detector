"""
data_prep.py — Dataset Analysis + Preprocessing Pipeline
=========================================================
Run modes:
    python data_prep.py --mode analyze       # Phase 1: EDA + stats (shows output for review)
    python data_prep.py --mode build_splits  # Phase 3: Build train/val/test splits
    python data_prep.py --mode inspect_hin   # Inspect hin-tagged samples with Whisper

Usage:
    python data_prep.py --mode analyze --sample_size 10000
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
STATS_DIR = BASE_DIR / "stats"
PLOTS_DIR = STATS_DIR / "plots"
CACHE_DIR = BASE_DIR / "cache"
SPLITS_DIR = BASE_DIR / "splits"
HIN_SAMPLES_DIR = BASE_DIR / "hin_samples"

for d in [STATS_DIR, PLOTS_DIR, CACHE_DIR, SPLITS_DIR, HIN_SAMPLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
DATASET_NAME = "pipecat-ai/smart-turn-data-v3.2-train"
TARGET_SR = 16_000
TAIL_WINDOW_SEC = 2.0   # seconds of tail to use (ablated in train.py)
FEATURE_HOP_MS = 10     # Whisper hop: 10ms per frame → 3000 frames for 30s
NUM_MEL_BINS = 80


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 1: ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def run_analysis(sample_size: int = 15_000):
    """Stream-sample the dataset and produce all distribution stats."""
    print("\n" + "="*70)
    print("PHASE 1: DATA ANALYSIS")
    print("="*70)
    print(f"Streaming up to {sample_size:,} rows from {DATASET_NAME} ...")

    from datasets import load_dataset

    ds = load_dataset(DATASET_NAME, split="train", streaming=True)

    records = []
    for i, row in enumerate(tqdm(ds, total=sample_size, desc="Sampling")):
        if i >= sample_size:
            break
        audio = row.get("audio", {})
        audio_array = audio.get("array", None) if isinstance(audio, dict) else None
        duration = len(audio_array) / audio.get("sampling_rate", TARGET_SR) if audio_array is not None else None

        records.append({
            "id":           row.get("id", i),
            "language":     row.get("language", "unk"),
            "endpoint_bool": bool(row.get("endpoint_bool", False)),
            "midfiller":    bool(row.get("midfiller", False)),
            "endfiller":    bool(row.get("endfiller", False)),
            "synthetic":    bool(row.get("synthetic", False)),
            "dataset":      row.get("dataset", "unk"),
            "has_text":     row.get("spoken_text") is not None,
            "duration_sec": duration,
        })

    df = pd.DataFrame(records)
    print(f"\nSampled {len(df):,} rows successfully.")
    print(f"Columns: {list(df.columns)}")

    # ── 1. Overall class balance ──────────────────────────────────────────────
    print("\n" + "-"*50)
    print("1. OVERALL CLASS BALANCE (endpoint_bool)")
    print("-"*50)
    bal = df["endpoint_bool"].value_counts()
    bal_pct = df["endpoint_bool"].value_counts(normalize=True) * 100
    for v in [True, False]:
        print(f"  endpoint_bool={v}: {bal[v]:>6,}  ({bal_pct[v]:.1f}%)")
    pos_weight = bal[False] / bal[True]
    print(f"  → pos_weight for BCEWithLogitsLoss: {pos_weight:.3f}")

    # ── 2. Language distribution ──────────────────────────────────────────────
    print("\n" + "-"*50)
    print("2. LANGUAGE DISTRIBUTION (top 15)")
    print("-"*50)
    lang_counts = df["language"].value_counts()
    for lang, cnt in lang_counts.head(15).items():
        hin_mark = " ← TARGET (hin)" if lang == "hin" else ""
        print(f"  {lang:8s}: {cnt:>6,}  ({100*cnt/len(df):.1f}%){hin_mark}")

    # ── 3. Class balance PER language (top 10) ───────────────────────────────
    print("\n" + "-"*50)
    print("3. endpoint_bool=True RATE PER LANGUAGE (top 10 langs)")
    print("-"*50)
    top_langs = lang_counts.head(10).index.tolist()
    if "hin" not in top_langs:
        top_langs.append("hin")
    for lang in top_langs:
        sub = df[df["language"] == lang]
        if len(sub) == 0:
            print(f"  {lang:8s}: NO SAMPLES")
            continue
        rate = sub["endpoint_bool"].mean() * 100
        print(f"  {lang:8s}: n={len(sub):>5,}  true_rate={rate:.1f}%")

    # ── 4. Dataset-source distribution ───────────────────────────────────────
    print("\n" + "-"*50)
    print("4. DATASET SOURCE DISTRIBUTION (leakage check)")
    print("-"*50)
    src_counts = df["dataset"].value_counts()
    for src, cnt in src_counts.items():
        sub = df[df["dataset"] == src]
        true_rate = sub["endpoint_bool"].mean() * 100
        print(f"  {src:20s}: n={cnt:>5,}  true_rate={true_rate:.1f}%")

    # ── 5. Hard negatives ────────────────────────────────────────────────────
    print("\n" + "-"*50)
    print("5. HARD NEGATIVES (midfiller=True AND endpoint_bool=False)")
    print("-"*50)
    hard_neg = df[(df["midfiller"]) & (~df["endpoint_bool"])]
    easy_neg = df[(~df["midfiller"]) & (~df["endpoint_bool"])]
    all_neg = df[~df["endpoint_bool"]]
    print(f"  All negatives:        {len(all_neg):>6,}")
    print(f"  Hard negatives (mid): {len(hard_neg):>6,}  ({100*len(hard_neg)/max(1,len(all_neg)):.1f}% of negatives)")
    print(f"  Easy negatives:       {len(easy_neg):>6,}  ({100*len(easy_neg)/max(1,len(all_neg)):.1f}% of negatives)")
    print(f"  → These {len(hard_neg):,} hard negatives will be 3× oversampled in Experiment 3")

    # ── 6. Audio duration ────────────────────────────────────────────────────
    print("\n" + "-"*50)
    print("6. AUDIO DURATION DISTRIBUTION")
    print("-"*50)
    dur = df["duration_sec"].dropna()
    if len(dur) > 0:
        print(f"  Mean:   {dur.mean():.2f}s")
        print(f"  Median: {dur.median():.2f}s")
        print(f"  p5:     {dur.quantile(0.05):.2f}s")
        print(f"  p95:    {dur.quantile(0.95):.2f}s")
        print(f"  Min:    {dur.min():.2f}s")
        print(f"  Max:    {dur.max():.2f}s")
        print(f"  → Using last {TAIL_WINDOW_SEC}s tail window (covers p5={dur.quantile(0.05):.1f}s)")
    else:
        print("  Duration data not available in sample")

    # ── 7. Synthetic vs real ─────────────────────────────────────────────────
    print("\n" + "-"*50)
    print("7. SYNTHETIC vs REAL SPEECH")
    print("-"*50)
    synth = df["synthetic"].value_counts()
    for v, cnt in synth.items():
        print(f"  synthetic={v}: {cnt:>6,}  ({100*cnt/len(df):.1f}%)")

    # ── 8. spoken_text availability ──────────────────────────────────────────
    print("\n" + "-"*50)
    print("8. SPOKEN TEXT AVAILABILITY")
    print("-"*50)
    has_text = df["has_text"].sum()
    print(f"  Rows with spoken_text: {has_text:,} / {len(df):,}  ({100*has_text/len(df):.1f}%)")
    print("  → Do NOT rely on spoken_text in the pipeline (too sparse)")

    # ── HIN SUBSET CHECK ──────────────────────────────────────────────────────
    print("\n" + "-"*50)
    print("9. HIN-TAGGED SUBSET ANALYSIS")
    print("-"*50)
    hin_df = df[df["language"] == "hin"]
    if len(hin_df) == 0:
        print("  ⚠ No 'hin' samples in this sample batch — try increasing --sample_size")
    else:
        print(f"  hin rows in sample: {len(hin_df)}")
        print(f"  endpoint_bool distribution: {dict(hin_df['endpoint_bool'].value_counts())}")
        print(f"  synthetic distribution:      {dict(hin_df['synthetic'].value_counts())}")
        print(f"  dataset sources:             {dict(hin_df['dataset'].value_counts())}")
        print(f"  has spoken_text:             {hin_df['has_text'].sum()}/{len(hin_df)}")
        hin_dur = hin_df["duration_sec"].dropna()
        if len(hin_dur) > 0:
            print(f"  duration mean/median:        {hin_dur.mean():.2f}s / {hin_dur.median():.2f}s")
        print()
        print("  ⚠ IMPORTANT: 'hin' tag likely = monolingual Hindi TTS/read speech,")
        print("    NOT code-switched Hinglish. Run --mode inspect_hin to confirm with")
        print("    Whisper transcription. A separate Hinglish eval set is REQUIRED.")

    # ── SAVE STATS ────────────────────────────────────────────────────────────
    stats_dict = {
        "sample_size": len(df),
        "overall_balance": {
            "true":  int(bal.get(True, 0)),
            "false": int(bal.get(False, 0)),
            "true_pct": float(bal_pct.get(True, 0)),
            "false_pct": float(bal_pct.get(False, 0)),
            "pos_weight": float(pos_weight),
        },
        "language_counts": lang_counts.to_dict(),
        "dataset_source_counts": src_counts.to_dict(),
        "hard_negatives": {
            "count": int(len(hard_neg)),
            "pct_of_negatives": float(100*len(hard_neg)/max(1,len(all_neg))),
        },
        "hin_subset": {
            "count": int(len(hin_df)),
            "endpoint_true": int((hin_df["endpoint_bool"]).sum()) if len(hin_df) > 0 else 0,
        },
        "duration_stats": {
            "mean": float(dur.mean()) if len(dur) > 0 else None,
            "median": float(dur.median()) if len(dur) > 0 else None,
            "p5": float(dur.quantile(0.05)) if len(dur) > 0 else None,
            "p95": float(dur.quantile(0.95)) if len(dur) > 0 else None,
        },
    }
    stats_path = STATS_DIR / "data_analysis.json"
    with open(stats_path, "w") as f:
        json.dump(stats_dict, f, indent=2)
    print(f"\n✓ Stats saved to {stats_path}")

    # ── PLOTS ─────────────────────────────────────────────────────────────────
    _make_plots(df, lang_counts)
    print(f"✓ Plots saved to {PLOTS_DIR}/")
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE — review above before running --mode build_splits")
    print("="*70)
    return df, stats_dict


def _make_plots(df: pd.DataFrame, lang_counts: pd.Series):
    """Save all EDA plots."""
    sns.set_theme(style="darkgrid", palette="muted")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Smart Turn Dataset — EDA Summary", fontsize=16, fontweight="bold")

    # 1. Class balance
    ax = axes[0, 0]
    vals = df["endpoint_bool"].value_counts()
    ax.bar(["False\n(mid-turn)", "True\n(turn-end)"],
           [vals.get(False, 0), vals.get(True, 0)],
           color=["#e74c3c", "#2ecc71"])
    ax.set_title("Class Balance (endpoint_bool)")
    ax.set_ylabel("Count")

    # 2. Top languages
    ax = axes[0, 1]
    top = lang_counts.head(12)
    ax.barh(top.index[::-1], top.values[::-1], color="#3498db")
    ax.set_title("Top 12 Languages by Row Count")
    ax.set_xlabel("Count")

    # 3. True-rate per top language
    ax = axes[0, 2]
    top_langs = lang_counts.head(10).index.tolist()
    rates = [df[df["language"] == l]["endpoint_bool"].mean() * 100 for l in top_langs]
    colors = ["#e74c3c" if r < 45 else "#2ecc71" if r > 55 else "#f39c12" for r in rates]
    ax.barh(top_langs[::-1], rates[::-1], color=colors[::-1])
    ax.axvline(50, color="white", linestyle="--", alpha=0.7)
    ax.set_title("True Rate % per Language (top 10)")
    ax.set_xlabel("% endpoint_bool=True")

    # 4. Dataset source leakage
    ax = axes[1, 0]
    srcs = df["dataset"].value_counts()
    true_rates = [df[df["dataset"] == s]["endpoint_bool"].mean() * 100 for s in srcs.index]
    scatter = ax.scatter(srcs.values, true_rates, c=true_rates, cmap="RdYlGn",
                         s=100, vmin=30, vmax=70)
    for s, cnt, rate in zip(srcs.index, srcs.values, true_rates):
        ax.annotate(s, (cnt, rate), fontsize=7, ha="center", va="bottom")
    ax.set_title("Source Leakage Check\n(x=count, y=true_rate%)")
    ax.set_xlabel("Row count per source")
    ax.set_ylabel("endpoint_bool=True %")
    plt.colorbar(scatter, ax=ax)

    # 5. Duration distribution
    ax = axes[1, 1]
    dur = df["duration_sec"].dropna()
    if len(dur) > 0:
        ax.hist(dur, bins=50, color="#9b59b6", alpha=0.85)
        ax.axvline(TAIL_WINDOW_SEC, color="orange", linestyle="--",
                   label=f"Tail window {TAIL_WINDOW_SEC}s")
        ax.set_title("Audio Duration Distribution")
        ax.set_xlabel("Duration (seconds)")
        ax.legend()

    # 6. Hard negatives breakdown
    ax = axes[1, 2]
    neg_types = {
        "Hard neg\n(midfiller=T)": int(((df["midfiller"]) & (~df["endpoint_bool"])).sum()),
        "Easy neg\n(midfiller=F)": int(((~df["midfiller"]) & (~df["endpoint_bool"])).sum()),
        "Positives": int(df["endpoint_bool"].sum()),
    }
    ax.bar(neg_types.keys(), neg_types.values(),
           color=["#e74c3c", "#e67e22", "#2ecc71"])
    ax.set_title("Hard vs Easy Negatives")
    ax.set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "eda_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {PLOTS_DIR / 'eda_summary.png'}")


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 1b: INSPECT HIN SAMPLES WITH WHISPER
# ──────────────────────────────────────────────────────────────────────────────

def inspect_hin_samples(n_samples: int = 10):
    """
    Download n_samples of hin-tagged audio, save them, and transcribe
    with Whisper base to confirm monolingual Hindi vs code-switched Hinglish.
    This is for INSPECTION ONLY — not part of the training pipeline.
    """
    print("\n" + "="*70)
    print("PHASE 1b: INSPECTING HIN-TAGGED SAMPLES WITH WHISPER")
    print("="*70)

    import soundfile as sf
    import whisper
    from datasets import load_dataset

    print(f"Loading Whisper base for transcription inspection...")
    w_model = whisper.load_model("base")

    print(f"Streaming dataset to find hin-tagged samples...")
    ds = load_dataset(DATASET_NAME, split="train", streaming=True)

    found = 0
    results = []
    for row in tqdm(ds, desc="Scanning for hin"):
        if row.get("language") != "hin":
            continue
        if found >= n_samples:
            break

        audio = row["audio"]
        array = np.array(audio["array"], dtype=np.float32)
        sr = audio["sampling_rate"]

        # Save wav
        wav_path = HIN_SAMPLES_DIR / f"hin_{found:03d}.wav"
        sf.write(str(wav_path), array, sr)

        # Transcribe
        result = w_model.transcribe(str(wav_path), language=None)
        transcript = result["text"].strip()
        detected_lang = result.get("language", "?")

        row_info = {
            "idx": found,
            "id": row.get("id", "?"),
            "endpoint_bool": bool(row.get("endpoint_bool")),
            "synthetic": bool(row.get("synthetic")),
            "dataset_src": row.get("dataset", "?"),
            "transcript": transcript,
            "whisper_detected_lang": detected_lang,
            "duration_sec": len(array) / sr,
        }
        results.append(row_info)

        print(f"\n  [{found}] endpoint={row_info['endpoint_bool']} | src={row_info['dataset_src']} | lang_detected={detected_lang}")
        print(f"       transcript: {transcript[:200]}")
        found += 1

    # Summarise
    print("\n" + "-"*50)
    print("HIN INSPECTION SUMMARY:")
    detected_langs = Counter(r["whisper_detected_lang"] for r in results)
    print(f"  Whisper-detected languages: {dict(detected_langs)}")
    code_switched = sum(1 for r in results if any(
        eng_word in r["transcript"].lower()
        for eng_word in ["the", "is", "are", "to", "and", "you", "order", "delivery"]
    ))
    print(f"  Clips with obvious English words (code-switch indicator): {code_switched}/{len(results)}")
    print()
    print("  ⚠ If whisper_detected_lang is predominantly 'hi' (Hindi) and")
    print("    transcripts show no English code-switching, this confirms the hin")
    print("    tag = monolingual Hindi TTS, NOT Hinglish. A supplementary")
    print("    Hinglish eval set is required (see hinglish_eval_builder.py).")

    # Save results
    results_path = HIN_SAMPLES_DIR / "inspection_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Saved {len(results)} inspection results to {results_path}")


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 3: BUILD SPLITS
# ──────────────────────────────────────────────────────────────────────────────

def build_splits(
    n_train: int = 5000,
    n_val:   int = 1000,
    n_test:  int = 1000,
    tail_sec: float = TAIL_WINDOW_SEC,
    hard_neg_oversample: int = 3,
    seed: int = 42,
):
    """
    Stream the full dataset and build preprocessed splits.
    Stratifies by (language, endpoint_bool) jointly.
    Hard negatives (midfiller=True & endpoint_bool=False) are oversampled
    in training by hard_neg_oversample x.
    """
    print("\n" + "="*70)
    print("PHASE 3: BUILDING TRAIN / VAL / TEST SPLITS")
    print("="*70)
    print(f"  n_train={n_train:,}, n_val={n_val:,}, n_test={n_test:,}")
    print(f"  Tail window: {tail_sec}s")
    print(f"  Hard-neg oversample: {hard_neg_oversample}x")

    from datasets import load_dataset
    from transformers import WhisperFeatureExtractor
    import soundfile as sf
    import librosa

    extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")
    rng = np.random.default_rng(seed)

    def preprocess_audio(array, sr):
        """Resample to 16kHz, extract tail, extract features."""
        if sr != TARGET_SR:
            array = librosa.resample(array, orig_sr=sr, target_sr=TARGET_SR)
        # Extract tail
        tail_samples = int(tail_sec * TARGET_SR)
        if len(array) > tail_samples:
            array = array[-tail_samples:]
        else:
            pad = tail_samples - len(array)
            array = np.pad(array, (pad, 0), mode="constant")
        array = array.astype(np.float32)
        # WhisperFeatureExtractor expects max 30s → we pass 2s but it zero-pads
        feats = extractor(array, sampling_rate=TARGET_SR, return_tensors="np")
        return feats.input_features[0]  # (80, 3000)

    ds = load_dataset(DATASET_NAME, split="train", streaming=True)

    records = []
    total = n_train + n_val + n_test
    print(f"  Streaming {total:,} rows...")
    for i, row in enumerate(tqdm(ds, total=total)):
        if i >= total:
            break
        try:
            audio = row["audio"]
            array = np.array(audio["array"], dtype=np.float32)
            sr = audio["sampling_rate"]
            feats = preprocess_audio(array, sr)
            records.append({
                "features": feats,
                "label":    int(bool(row.get("endpoint_bool", False))),
                "language": row.get("language", "unk"),
                "dataset_src": row.get("dataset", "unk"),
                "midfiller": bool(row.get("midfiller", False)),
                "endfiller": bool(row.get("endfiller", False)),
                "synthetic": bool(row.get("synthetic", False)),
            })
        except Exception as e:
            continue

    df = pd.DataFrame([{k: v for k, v in r.items() if k != "features"} for r in records])
    features = [r["features"] for r in records]

    print(f"  Processed {len(df):,} rows")
    print(f"  Label balance: {dict(df['label'].value_counts())}")

    # Stratified split by (language, label)
    from sklearn.model_selection import train_test_split

    strat_key = df["language"].str[:4] + "_" + df["label"].astype(str)
    idx_all = np.arange(len(df))

    # First carve out test set
    idx_trainval, idx_test = train_test_split(
        idx_all, test_size=n_test, stratify=strat_key,
        random_state=seed
    )
    strat_trainval = strat_key.iloc[idx_trainval].reset_index(drop=True)
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=n_val, stratify=strat_trainval,
        random_state=seed
    )

    def save_split(indices, name, oversample_hard_neg=False):
        sub = df.iloc[indices].copy()
        sub_features = [features[i] for i in indices]

        if oversample_hard_neg and hard_neg_oversample > 1:
            hard_mask = (sub["midfiller"]) & (sub["label"] == 0)
            hard_idx = sub.index[hard_mask].tolist()
            # Repeat hard negatives
            extra_idx = hard_idx * (hard_neg_oversample - 1)
            extra_feats = [features[i] for i in extra_idx]
            extra_df = sub.loc[extra_idx].reset_index(drop=True)

            sub = pd.concat([sub, extra_df], ignore_index=True)
            sub_features = sub_features + extra_feats

            print(f"  [{name}] Hard-neg oversample: {len(hard_idx)} → {len(hard_idx)*hard_neg_oversample} samples added")

        # Shuffle
        shuffle_idx = rng.permutation(len(sub))
        sub = sub.iloc[shuffle_idx].reset_index(drop=True)
        sub_features = [sub_features[i] for i in shuffle_idx]

        # Save
        out_dir = SPLITS_DIR / name
        out_dir.mkdir(exist_ok=True)
        np.save(str(out_dir / "features.npy"), np.array(sub_features, dtype=np.float32))
        sub.drop(columns=[], errors="ignore").to_parquet(str(out_dir / "metadata.parquet"))
        print(f"  [{name}] Saved {len(sub):,} rows → label balance: {dict(sub['label'].value_counts())}")

    save_split(idx_train, "train", oversample_hard_neg=False)  # Exp 1 & 2
    save_split(idx_train, "train_hn", oversample_hard_neg=True)  # Exp 3 (hard-neg oversampled)
    save_split(idx_val,   "val",   oversample_hard_neg=False)
    save_split(idx_test,  "test",  oversample_hard_neg=False)

    print(f"\n✓ All splits saved to {SPLITS_DIR}/")
    print("  Next: python train.py --experiment 1|2|3")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Data preparation pipeline")
    parser.add_argument("--mode", choices=["analyze", "build_splits", "inspect_hin"],
                        default="analyze")
    parser.add_argument("--sample_size", type=int, default=15_000,
                        help="Rows to sample for analysis (default 15000)")
    args = parser.parse_args()

    if args.mode == "analyze":
        run_analysis(sample_size=args.sample_size)
    elif args.mode == "inspect_hin":
        inspect_hin_samples()
    elif args.mode == "build_splits":
        build_splits()
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
