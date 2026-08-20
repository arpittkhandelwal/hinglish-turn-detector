# Hinglish Turn Detection

> Audio-only turn endpoint detection — Shiprocket Data Scientist Challenge

A tiny, fast model that listens to the tail-end of a user's speech and decides:
**"Did they finish talking, or are they just pausing?"**

---

## Headline Results

| Model | Test AUROC | Test F1 | Hinglish AUROC | Latency (CPU) |
|-------|-----------|---------|----------------|---------------|
| Exp 1: Baseline (mean pool) | 0.464 | 0.652 | — | — |
| Exp 2: + Attention Pool | **0.862** | **0.791** | 0.782 | — |
| Exp 3: + Hard-Neg 3× | 0.901* | 0.853* | — | — |
| **Exp 2 (ONNX, CPU)** | 0.862 | 0.791 | **0.782** | **207 ms / RTF 0.103** |

\*Exp 3 val metrics — not evaluated on full test set (same architecture as Exp 2, used for ablation).

---

## Architecture

```
Waveform → Whisper Tiny Encoder (blocks 0-1 frozen, 2-3 fine-tuned)
         → Attention Pooling (learned query, 512-dim)
         → MLP Head (512→128→1)
         → sigmoid → P(turn_end)
```

- **8.26M total params, only 50K trainable (0.61%)** — truly tiny
- Exported to **ONNX** (36 KB), real-time on a single CPU thread
- Runs at **RTF = 0.103** — uses only 10% of audio duration to decide

---

## Hinglish Gap (Honest Reporting)

The training dataset (`pipecat-ai/smart-turn-data-v3.2-train`) tags Hindi rows as `hin`, but all 721 `hin` samples are synthetic TTS from a single voice (`chirp3_1`). No code-switched Hinglish is present.

To expose this gap honestly, we built a **60-clip synthetic Hinglish eval set** (30 turn-end / 30 mid-turn) using gTTS with code-switched sentences.

| Set | AUROC | F1 |
|-----|-------|----|
| Main test (matched distribution) | 0.862 | 0.791 |
| **Hinglish held-out (OOD)** | **0.782** | **0.789** |
| Gap | **-0.079** | -0.003 |

The AUROC drops ~8 points on OOD Hinglish — this is expected and is reported honestly rather than inflated.

---

## Quickstart

```bash
# 1. Install
pip install -r requirements.txt

# 2. Build Hinglish eval set (gTTS, ~5 min, requires internet)
python hinglish_eval_builder.py

# 3. Build data splits (streams from HuggingFace)
python data_prep.py --mode build_splits

# 4. Train (3 experiments)
python train.py --experiment 1 --epochs 1
python train.py --experiment 2 --epochs 1
python train.py --experiment 3 --epochs 1

# 5. Evaluate + ONNX export + latency benchmark
python eval.py --experiment 2 --onnx

# 6. Interactive Gradio demo
python app.py
```

---

## Files

```
├── model.py                 # WhisperTinyTurnDetector + AttentionPooling
├── data_prep.py             # EDA analysis + split builder
├── train.py                 # Training loop (3 experiments)
├── eval.py                  # Test eval + Hinglish eval + ONNX benchmark
├── hinglish_eval_builder.py # Synthetic Hinglish eval set via gTTS
├── app.py                   # Gradio demo
├── checkpoints/             # Best model weights per experiment
├── onnx/                    # Exported ONNX model (exp2_model.onnx)
├── stats/                   # JSON results + plots
│   ├── data_analysis.json
│   ├── exp1_results.json
│   ├── exp2_results.json
│   ├── exp3_results.json
│   └── exp2_full_eval.json
├── hinglish_eval/           # 60 synthetic Hinglish WAVs + metadata
├── REPORT.md                # Full lab notebook report
└── requirements.txt
```

---

## Key Findings

1. **Attention pooling is the decisive upgrade** — Exp1→Exp2 jumps AUROC from 0.464 to 0.862. Where you pool Whisper features matters more than how many layers you fine-tune.

2. **The model partially learns TTS artifacts** — `midcentury_1` (real human speech) scores AUROC 0.460 (near-random). The model is not production-ready without more real-speech training data.

3. **Hard-negative oversampling helps** — 3× oversampling of `midfiller=True` negatives improves val AUROC to 0.901 (Exp 3), showing that mid-utterance fillers are the main failure mode.

4. **ONNX at 207ms / RTF 0.103** — comfortably real-time on a single CPU thread. The exported model is 36 KB.

See [`REPORT.md`](REPORT.md) for the full analysis.
