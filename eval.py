"""
eval.py — Full Evaluation Pipeline
====================================
Runs:
  1. Main test set metrics (acc, precision, recall, F1, AUROC, confusion matrix)
  2. Per-language breakdown table
  3. Per-dataset-source breakdown (leakage check)
  4. Hinglish held-out set metrics (vs main test set)
  5. ONNX export + latency benchmark (CPU ms/inference, RTF)
  6. Error analysis (10-15 misclassified examples)

Usage:
    python eval.py --experiment 2            # evaluate best exp2 checkpoint
    python eval.py --experiment 2 --onnx     # also export+benchmark ONNX
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
)
from tqdm import tqdm

from model import build_model, WhisperTinyTurnDetector

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
SPLITS_DIR    = BASE_DIR / "splits"
CKPT_DIR      = BASE_DIR / "checkpoints"
STATS_DIR     = BASE_DIR / "stats"
PLOTS_DIR     = STATS_DIR / "plots"
HINGLISH_DIR  = BASE_DIR / "hinglish_eval"
ONNX_DIR      = BASE_DIR / "onnx"

for d in [STATS_DIR, PLOTS_DIR, ONNX_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def load_checkpoint(experiment: int, device: torch.device):
    ckpt_path = CKPT_DIR / f"exp{experiment}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Run: python train.py --experiment {experiment}"
        )
    ckpt = torch.load(str(ckpt_path), map_location=device)
    model = build_model(experiment).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Loaded checkpoint: {ckpt_path} (best epoch={ckpt['epoch']})")
    print(f"  Val metrics at checkpoint: {ckpt['val_metrics']}")
    return model, ckpt


def compute_metrics(labels, probs, prefix=""):
    preds = (np.array(probs) >= 0.5).astype(int)
    labels = np.array(labels)
    return {
        f"{prefix}accuracy":  float(accuracy_score(labels, preds)),
        f"{prefix}precision": float(precision_score(labels, preds, zero_division=0)),
        f"{prefix}recall":    float(recall_score(labels, preds, zero_division=0)),
        f"{prefix}f1":        float(f1_score(labels, preds, zero_division=0)),
        f"{prefix}auroc":     float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else 0.5,
        f"{prefix}n":         int(len(labels)),
    }


def run_inference(model, features_np: np.ndarray, batch_size: int = 64,
                   device: torch.device = torch.device("cpu")) -> np.ndarray:
    """Run inference on a numpy features array; returns probabilities."""
    all_probs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(features_np), batch_size):
            batch = torch.from_numpy(features_np[i:i+batch_size]).float().to(device)
            logits = model(batch).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
    return np.array(all_probs)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN EVALUATION
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_main_test(model, device, experiment: int) -> dict:
    print("\n" + "="*65)
    print("1. MAIN TEST SET EVALUATION")
    print("="*65)

    test_dir = SPLITS_DIR / "test"
    if not test_dir.exists():
        print("  ⚠ test split not found. Run: python data_prep.py --mode build_splits")
        return {}

    features = np.load(str(test_dir / "features.npy"), mmap_mode="r")
    meta = pd.read_parquet(str(test_dir / "metadata.parquet"))

    probs = run_inference(model, np.array(features), device=device)
    labels = meta["label"].values
    preds = (probs >= 0.5).astype(int)

    metrics = compute_metrics(labels, probs)
    print(f"\n  n={metrics['n']:,}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  AUROC:     {metrics['auroc']:.4f}")

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    print(f"\n  Confusion Matrix:")
    print(f"                  Pred=False  Pred=True")
    print(f"  Actual=False    {cm[0,0]:>8}   {cm[0,1]:>8}")
    print(f"  Actual=True     {cm[1,0]:>8}   {cm[1,1]:>8}")

    _save_confusion_matrix(cm, experiment, "test")

    # Per-language breakdown
    lang_metrics = _per_language_breakdown(meta, probs, labels)

    # Per-dataset-source breakdown
    source_metrics = _per_source_breakdown(meta, probs, labels)

    return {**metrics, "per_language": lang_metrics, "per_source": source_metrics}


def _per_language_breakdown(meta: pd.DataFrame, probs: np.ndarray, labels: np.ndarray) -> dict:
    print("\n" + "-"*55)
    print("  PER-LANGUAGE BREAKDOWN (top langs + hin)")
    print("-"*55)

    lang_counts = meta["language"].value_counts()
    top_langs = lang_counts.head(6).index.tolist()
    if "hin" not in top_langs:
        top_langs.append("hin")

    lang_metrics = {}
    rows = []
    for lang in top_langs:
        mask = meta["language"] == lang
        if mask.sum() == 0:
            continue
        m = compute_metrics(labels[mask], probs[mask])
        lang_metrics[lang] = m
        rows.append({
            "language": lang, "n": m["n"],
            "accuracy": m["accuracy"], "f1": m["f1"], "auroc": m["auroc"],
        })
        hin_flag = " ← hin" if lang == "hin" else ""
        print(f"  {lang:8s}: n={m['n']:>5,}  acc={m['accuracy']:.3f}  f1={m['f1']:.3f}  auroc={m['auroc']:.3f}{hin_flag}")

    return lang_metrics


def _per_source_breakdown(meta: pd.DataFrame, probs: np.ndarray, labels: np.ndarray) -> dict:
    print("\n" + "-"*55)
    print("  PER-DATASET-SOURCE BREAKDOWN (leakage check)")
    print("-"*55)

    src_metrics = {}
    for src in meta["dataset_src"].unique():
        mask = meta["dataset_src"] == src
        if mask.sum() < 10:
            continue
        m = compute_metrics(labels[mask], probs[mask])
        src_metrics[src] = m
        print(f"  {src:20s}: n={m['n']:>5,}  acc={m['accuracy']:.3f}  f1={m['f1']:.3f}  auroc={m['auroc']:.3f}")

    auroc_values = [v["auroc"] for v in src_metrics.values()]
    if auroc_values:
        delta = max(auroc_values) - min(auroc_values)
        print(f"\n  AUROC spread across sources: {delta:.3f}")
        if delta > 0.1:
            print("  ⚠ WARNING: Large AUROC spread — model may be learning source artifacts!")
        else:
            print("  ✓ AUROC spread is small — no obvious source leakage")

    return src_metrics


def _save_confusion_matrix(cm: np.ndarray, experiment: int, split_name: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Pred=False", "Pred=True"],
                yticklabels=["Actual=False", "Actual=True"])
    ax.set_title(f"Experiment {experiment} — {split_name} Confusion Matrix")
    plt.tight_layout()
    path = PLOTS_DIR / f"exp{experiment}_{split_name}_cm.png"
    plt.savefig(str(path), dpi=130)
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# HINGLISH HELD-OUT EVALUATION
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_hinglish(model, device, experiment: int) -> dict:
    print("\n" + "="*65)
    print("2. HINGLISH HELD-OUT SET EVALUATION")
    print("="*65)

    meta_path = HINGLISH_DIR / "metadata.csv"
    feats_path = HINGLISH_DIR / "features.npy"

    if not meta_path.exists() or not feats_path.exists():
        print("  ⚠ Hinglish eval set not found.")
        print("    Run: python hinglish_eval_builder.py")
        return {}

    features = np.load(str(feats_path))
    meta = pd.read_csv(str(meta_path))
    labels = meta["label"].values
    probs = run_inference(model, features, device=device)
    preds = (probs >= 0.5).astype(int)

    metrics = compute_metrics(labels, probs, prefix="hinglish_")
    print(f"\n  n={metrics['hinglish_n']}")
    print(f"  Accuracy:  {metrics['hinglish_accuracy']:.4f}")
    print(f"  Precision: {metrics['hinglish_precision']:.4f}")
    print(f"  Recall:    {metrics['hinglish_recall']:.4f}")
    print(f"  F1:        {metrics['hinglish_f1']:.4f}")
    print(f"  AUROC:     {metrics['hinglish_auroc']:.4f}")

    cm = confusion_matrix(labels, preds)
    _save_confusion_matrix(cm, experiment, "hinglish")
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# ONNX EXPORT + LATENCY BENCHMARK
# ──────────────────────────────────────────────────────────────────────────────

def export_and_benchmark_onnx(model: WhisperTinyTurnDetector, experiment: int) -> dict:
    print("\n" + "="*65)
    print("3. ONNX EXPORT + CPU LATENCY BENCHMARK")
    print("="*65)

    import onnx
    import onnxruntime as ort

    onnx_path = ONNX_DIR / f"exp{experiment}_model.onnx"

    # Export
    print(f"  Exporting to ONNX: {onnx_path}")
    model_cpu = model.cpu().eval()
    dummy = torch.randn(1, 80, 3000)  # (1, mel_bins, frames)

    torch.onnx.export(
        model_cpu,
        dummy,
        str(onnx_path),
        input_names=["input_features"],
        output_names=["logits"],
        dynamic_axes={"input_features": {0: "batch_size"}},
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"  ✓ ONNX model exported ({onnx_path.stat().st_size / 1e6:.1f} MB)")

    # Verify
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print(f"  ✓ ONNX model check passed")

    # Latency benchmark — 100 runs on CPU
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1  # single-threaded for fair CPU benchmark
    sess = ort.InferenceSession(str(onnx_path), sess_options=sess_options,
                                 providers=["CPUExecutionProvider"])

    dummy_np = dummy.numpy()
    WARMUP = 10
    N_RUNS = 100
    AUDIO_SEC = 2.0  # we feed 2s clips

    # Warm up
    for _ in range(WARMUP):
        sess.run(None, {"input_features": dummy_np})

    # Benchmark
    latencies = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        sess.run(None, {"input_features": dummy_np})
        latencies.append((time.perf_counter() - t0) * 1000)  # ms

    latencies = sorted(latencies)
    mean_ms = float(np.mean(latencies))
    p50_ms  = float(np.percentile(latencies, 50))
    p95_ms  = float(np.percentile(latencies, 95))
    rtf     = float(mean_ms / (AUDIO_SEC * 1000))  # real-time factor

    print(f"\n  CPU Latency Benchmark ({N_RUNS} runs, 1 thread):")
    print(f"  Mean:   {mean_ms:.1f} ms")
    print(f"  P50:    {p50_ms:.1f} ms")
    print(f"  P95:    {p95_ms:.1f} ms")
    print(f"  RTF:    {rtf:.4f}  (lower is faster; <1.0 = real-time capable)")
    print(f"  Input:  {AUDIO_SEC}s audio clip")

    # Verify ONNX vs PyTorch agreement
    pt_out = torch.sigmoid(model_cpu(dummy)).item()
    onnx_out = float(torch.sigmoid(torch.tensor(
        sess.run(None, {"input_features": dummy_np})[0][0, 0]
    )))
    max_diff = abs(pt_out - onnx_out)
    print(f"\n  PyTorch output: {pt_out:.6f}")
    print(f"  ONNX output:    {onnx_out:.6f}")
    print(f"  Max diff:       {max_diff:.2e}  {'✓ OK' if max_diff < 1e-3 else '⚠ LARGE DIFF'}")

    # Optional int8 quantization
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        q_path = ONNX_DIR / f"exp{experiment}_model_int8.onnx"
        quantize_dynamic(str(onnx_path), str(q_path), weight_type=QuantType.QInt8)
        print(f"\n  ✓ INT8 quantized model saved ({q_path.stat().st_size / 1e6:.1f} MB)")

        sess_q = ort.InferenceSession(str(q_path), sess_options=sess_options,
                                       providers=["CPUExecutionProvider"])
        for _ in range(WARMUP):
            sess_q.run(None, {"input_features": dummy_np})
        q_latencies = []
        for _ in range(N_RUNS):
            t0 = time.perf_counter()
            sess_q.run(None, {"input_features": dummy_np})
            q_latencies.append((time.perf_counter() - t0) * 1000)
        q_mean = float(np.mean(q_latencies))
        print(f"  INT8 mean latency: {q_mean:.1f} ms  (vs FP32: {mean_ms:.1f} ms, {mean_ms/q_mean:.1f}x speedup)")
    except Exception as e:
        print(f"  INT8 quantization skipped: {e}")
        q_mean = None

    return {
        "onnx_path": str(onnx_path),
        "onnx_size_mb": float(onnx_path.stat().st_size / 1e6),
        "latency_mean_ms": mean_ms,
        "latency_p50_ms": p50_ms,
        "latency_p95_ms": p95_ms,
        "rtf": rtf,
        "pt_onnx_max_diff": float(max_diff),
        "int8_mean_ms": q_mean,
    }


# ──────────────────────────────────────────────────────────────────────────────
# ERROR ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def error_analysis(model, device, experiment: int, n_examples: int = 15) -> list:
    print("\n" + "="*65)
    print("4. ERROR ANALYSIS")
    print("="*65)

    test_dir = SPLITS_DIR / "test"
    if not test_dir.exists():
        print("  ⚠ test split not found.")
        return []

    features = np.load(str(test_dir / "features.npy"), mmap_mode="r")
    meta = pd.read_parquet(str(test_dir / "metadata.parquet"))
    probs = run_inference(model, np.array(features), device=device)
    labels = meta["label"].values
    preds = (probs >= 0.5).astype(int)

    wrong_mask = preds != labels
    wrong_indices = np.where(wrong_mask)[0]
    print(f"  Total misclassified: {wrong_mask.sum()} / {len(labels)}")

    # Sort by confidence of wrong prediction (most confidently wrong first)
    wrong_probs = np.abs(probs[wrong_indices] - 0.5)  # higher = more confidently wrong
    sort_order = np.argsort(-wrong_probs)
    top_wrong = wrong_indices[sort_order[:n_examples]]

    errors = []
    print(f"\n  Top {n_examples} most confidently misclassified:")
    print(f"  {'idx':>5} {'true':>6} {'pred':>6} {'prob':>6} {'lang':>5} {'src':>12} {'midfill':>8} {'synth':>6}")
    print("  " + "-"*70)

    for i, idx in enumerate(top_wrong):
        row = meta.iloc[idx]
        true_label = int(labels[idx])
        pred_label = int(preds[idx])
        prob = float(probs[idx])
        err_type = "FP" if pred_label == 1 and true_label == 0 else "FN"
        errors.append({
            "idx": int(idx),
            "true_label": true_label,
            "pred_label": pred_label,
            "prob": prob,
            "error_type": err_type,
            "language": row.get("language", "?"),
            "dataset_src": row.get("dataset_src", "?"),
            "midfiller": bool(row.get("midfiller", False)),
            "endfiller": bool(row.get("endfiller", False)),
            "synthetic": bool(row.get("synthetic", False)),
        })
        print(
            f"  {idx:>5} {true_label:>6} {pred_label:>6} {prob:>6.3f} "
            f"{row.get('language','?'):>5} {str(row.get('dataset_src','?')):>12} "
            f"{str(row.get('midfiller',False)):>8} {str(row.get('synthetic',False)):>6}"
        )

    # Summarise error patterns
    fp_errors = [e for e in errors if e["error_type"] == "FP"]
    fn_errors = [e for e in errors if e["error_type"] == "FN"]
    print(f"\n  False Positives (predicted turn-end, was mid-turn): {len(fp_errors)}")
    print(f"  False Negatives (predicted mid-turn, was turn-end): {len(fn_errors)}")

    if fp_errors:
        mid_fp = sum(1 for e in fp_errors if e["midfiller"])
        print(f"  → FP with midfiller=True: {mid_fp}/{len(fp_errors)}")
        print("    (model predicted turn-end on clips that had mid-utterance fillers)")

    if fn_errors:
        end_fn = sum(1 for e in fn_errors if e["endfiller"])
        print(f"  → FN with endfiller=True: {end_fn}/{len(fn_errors)}")
        print("    (model predicted mid-turn on genuine turn-ends with trailing fillers)")

    # Save errors
    err_path = STATS_DIR / f"exp{experiment}_errors.json"
    with open(err_path, "w") as f:
        json.dump(errors, f, indent=2)
    print(f"\n  ✓ Error analysis saved to {err_path}")
    return errors


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=int, choices=[1, 2, 3], default=2)
    parser.add_argument("--onnx", action="store_true", help="Export + benchmark ONNX")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating experiment {args.experiment} on {device}")

    model, ckpt = load_checkpoint(args.experiment, device)

    # Run all evaluations
    test_metrics    = evaluate_main_test(model, device, args.experiment)
    hinglish_metrics = evaluate_hinglish(model, device, args.experiment)
    errors          = error_analysis(model, device, args.experiment)
    latency_metrics = {}
    if args.onnx:
        latency_metrics = export_and_benchmark_onnx(model, args.experiment)

    # Compile full results
    full_results = {
        "experiment": args.experiment,
        "checkpoint_epoch": ckpt.get("epoch"),
        "test": test_metrics,
        "hinglish": hinglish_metrics,
        "latency": latency_metrics,
        "n_errors_analyzed": len(errors),
    }
    out_path = STATS_DIR / f"exp{args.experiment}_full_eval.json"
    with open(out_path, "w") as f:
        json.dump(full_results, f, indent=2, default=str)
    print(f"\n✓ Full eval results → {out_path}")

    # Hinglish vs main gap
    if test_metrics and hinglish_metrics:
        print("\n" + "="*65)
        print("SUMMARY: MAIN TEST vs HINGLISH HELD-OUT")
        print("="*65)
        for key in ["accuracy", "f1", "auroc"]:
            main_val = test_metrics.get(key, 0)
            hing_val = hinglish_metrics.get(f"hinglish_{key}", 0)
            delta = hing_val - main_val
            sign = "+" if delta >= 0 else ""
            print(f"  {key:10s}: main={main_val:.4f}  hinglish={hing_val:.4f}  delta={sign}{delta:.4f}")


if __name__ == "__main__":
    main()
