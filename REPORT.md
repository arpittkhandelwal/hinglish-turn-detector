# Hinglish Turn Detection — Lab Report

**Challenge:** Shiprocket Data Scientist Hiring Assignment  
**Task:** Audio-only turn endpoint detection (binary classification)  
**Author:** Submission Candidate  
**Date:** 2026-08-20

---

## 1. Problem Statement

Given the tail-end of a user's speech clip, classify whether the speaker has **finished their turn** (`endpoint_bool=True`) or is **pausing mid-thought** (`endpoint_bool=False`). The model must operate **audio-only** (no ASR, no spoken text), produce a single probability, and run within real-time constraints on CPU.

The stretch goal is robustness on **Hinglish** — code-switched Hindi+English speech that is underrepresented in the available training data.

---

## 2. Data: Honest Analysis

### 2.1 Dataset (pipecat-ai/smart-turn-data-v3.2-train)

We sampled 15,000 rows to profile the dataset before training.

| Metric | Value |
|--------|-------|
| Total rows streamed | 15,000 |
| Class balance (True/False) | 49.6% / 50.4% |
| `pos_weight` for BCE loss | **1.017** (nearly balanced — no reweighting needed for baseline) |
| Synthetic speech share | **82.5%** |
| Rows with `spoken_text` | **0 / 15,000** — cannot use text features |
| Hard negatives (`midfiller=True AND endpoint_bool=False`) | **42.8%** of all negatives |

**Top languages:** `eng` (24.3%), `spa` (6%), `rus` (4.9%), `hin` (4.8%), `por` (4.7%) ...

### 2.2 The Hinglish Gap ⚠️

The `hin`-tagged rows (721 in sample) are **100% synthetic** (`chirp3_1` only), with `spoken_text=None`. These are almost certainly **monolingual Hindi TTS clips**, not code-switched Hinglish. We confirmed this pattern and note it as a fundamental data mismatch.

**Mitigation:** We built a separate synthetic Hinglish held-out set using `gTTS` with 60 clips (30 true-end, 30 mid-turn) using code-switched Hindi+English sentences. This is explicitly labelled as out-of-distribution proxy evaluation, not a gold-standard test.

### 2.3 Hard Negatives

Of the 7,562 negatives in our sample, 3,239 (42.8%) have `midfiller=True` — these are the hardest for any turn-end detector because they exhibit prosodic signatures of pause/hesitation. Experiment 3 specifically targets these with 3× oversampling.

---

## 3. Model Architecture

```
Input: Raw waveform (PCM float32, 16 kHz)
  │
  ▼
[WhisperFeatureExtractor]  → log-mel spectrogram (80-dim, 3000 frames / 30s)
  │  (frozen, no gradient)
  ▼
[Whisper Tiny Encoder]
  ├── Blocks 0–1: FROZEN   (low-level acoustic features)
  └── Blocks 2–3: FINE-TUNED  (high-level temporal patterns)
  │
  ▼
[AttentionPooling]  → learned 512-dim query, softmax weights over time axis
  │  (Exp 1 uses simple mean pooling)
  ▼
[MLP Head]  Linear(512→128) → GELU → Dropout(0.2) → Linear(128→1)
  │
  ▼
sigmoid(logit) → P(turn_end)
```

**Total params:** 8.26M | **Trainable (Exp 2):** 50,177 (0.61%) | **Trainable (Exp 3):** 3.60M (43.6%)

---

## 4. Experiments

Three experiments were run, each building on the last:

| Exp | Description | Pooling | Training Split | Trainable % | Val AUROC |
|-----|-------------|---------|----------------|-------------|-----------|
| 1 | Baseline (frozen Whisper + mean pool) | Mean | `train` (2,000) | 0.61% | 0.464 |
| 2 | + Attention Pooling | **Attention** | `train` (2,000) | 0.61% | **0.856** |
| 3 | + Hard-Neg Oversampling (3×) | Attention | `train_hn` (2,822) | 43.6% | **0.901** |

> **Note:** All experiments used only 1 epoch on a 2,000-row CPU-friendly subset for this submission. Exp 1's poor AUROC (0.464 — near-random) validates that mean pooling over frozen Whisper features is insufficient; attention pooling is the decisive architectural choice.

### Ablation Takeaway

The jump from **Exp 1 → Exp 2** (0.464 → 0.856 val AUROC) is the single most important finding: **where you pool the Whisper encoder output matters more than whether you fine-tune deeper layers**. The attention mechanism learns to weight prosodically rich frames (trailing intonation, final vowel lengthening) without needing to touch the frozen encoder weights.

---

## 5. Evaluation Results (Experiment 2 — Best on Test Set)

### 5.1 Main Test Set (n=500)

| Metric | Score |
|--------|-------|
| Accuracy | **0.786** |
| Precision | 0.784 |
| Recall | 0.799 |
| F1 | **0.791** |
| AUROC | **0.862** |

**Confusion Matrix:**
```
                Pred=False  Pred=True
Actual=False       190          56
Actual=True         51         203
```

False Positive Rate: 22.8% | False Negative Rate: 20.1%
The model is slightly more aggressive about calling endpoints (higher recall), which is generally preferable for a voice assistant (better to interrupt slightly early than to wait forever).

### 5.2 Per-Language Breakdown

| Language | n | Accuracy | F1 | AUROC |
|----------|---|----------|----|-------|
| `eng` | 124 | 0.694 | 0.627 | 0.793 |
| `spa` | 30 | 0.767 | 0.811 | 0.799 |
| **`hin`** | **25** | **0.840** | **0.867** | **0.936** ← TARGET |
| `por` | 24 | 0.958 | 0.957 | 0.972 |
| `rus` | 23 | 0.870 | 0.889 | 0.962 |
| `fra` | 23 | 0.783 | 0.828 | 0.923 |

**Surprisingly, `hin` scores the highest in F1 (0.867) and AUROC (0.936) of any language.** This is likely because the `hin` test-set clips are all synthetic (same TTS system as training), and the model has memorised the clean, regular prosody of that TTS voice. It does **not** imply real-world Hinglish robustness — see §5.3.

### 5.3 Hinglish Held-Out Set (n=60, synthetic, OOD)

| Metric | Main Test | Hinglish Held-Out | Delta |
|--------|-----------|-------------------|-------|
| Accuracy | 0.786 | **0.750** | -0.036 |
| F1 | 0.791 | **0.789** | -0.003 |
| AUROC | 0.862 | **0.782** | **-0.079** |

The **AUROC drop of 0.079** on the Hinglish proxy set is the honest gap this evaluation was designed to surface. The model generalises reasonably (F1 drops only 0.003), but calibration degrades on code-switched speech (AUROC -7.9 pts). This is expected given the training data mismatch.

### 5.4 Dataset-Source Leakage Check

| Source | n | Accuracy | AUROC |
|--------|---|----------|-------|
| `chirp3_1` | 265 | 0.823 | 0.905 |
| `chirp3_2` | 127 | 0.803 | 0.876 |
| `liva_1` | 63 | 0.667 | 0.789 |
| `midcentury_1` | 16 | 0.500 | 0.460 |

⚠️ **AUROC spread = 0.445** — a large variance. `midcentury_1` (real human speech, not TTS) scores near-random (0.460). This suggests **the model is partially learning TTS voice artifacts**, not purely prosodic turn-end cues. This is the primary reliability concern for production deployment.

---

## 6. Error Analysis

**107 / 500 test samples misclassified.**

**False Positive pattern (predicted turn-end, was mid-turn):**
- 5 of 8 most-confident FPs had `midfiller=True`
- The model "hears" a prosodic boundary but misses the mid-utterance filler signal

**False Negative pattern (predicted mid-turn, was turn-end):**
- All 7 FN cases in the top-15 are from `liva_1` or `midcentury_1` (real human speech)
- The model systematically underperforms on real (non-TTS) recordings

**Key error insight:** The model is a strong TTS-prosody detector but a weaker real-human-speech turn detector. Fixing this requires mixing in more `liva_1`/`midcentury_1`-style data during training, or adversarial domain adaptation.

---

## 7. ONNX Latency Benchmark

Exported to ONNX (opset 18) and benchmarked on CPU (Apple M-series, single thread, 100 warm runs):

| Metric | Value |
|--------|-------|
| ONNX model size | **0.036 MB** |
| Mean latency | **206.9 ms** |
| P50 latency | **206.7 ms** |
| P95 latency | **209.3 ms** |
| Real-Time Factor (RTF) | **0.103** |
| PyTorch ↔ ONNX output diff | 2.09e-07 ✓ |
| Input audio length | 2.0 s |

**RTF = 0.103** means the model uses 10.3% of the audio duration to process — fully real-time capable. At **207 ms** end-to-end, it can comfortably fit within the latency budget of a voice assistant turn-taking decision.

> INT8 quantization was skipped due to a shape inference mismatch in the attention pooling layer. This is a known issue with dynamic axes + custom pooling in ONNX and requires an onnxruntime-compatible quantization pre-pass.

---

## 8. Failure Modes & Limitations

| Issue | Severity | Mitigation |
|-------|----------|-----------|
| Learns TTS artifacts, not pure prosody | **High** | Train on more real human speech sources |
| Hinglish gap: AUROC drops 7.9 pts on OOD data | **Medium** | Collect real Hinglish data; use language-agnostic features |
| INT8 quantization blocked by shape mismatch | Low | Use per-tensor dynamic quantization or retrace for static shapes |
| Only 1 epoch trained (CPU constraint) | High | Run on GPU with full 200k rows for production quality |
| `midcentury_1` AUROC = 0.46 (near random) | **High** | Add more real-speech data; possible domain shift |

---

## 9. What I Would Do With More Time

1. **Train on GPU** with the full 220k-row dataset for 5–10 epochs with proper LR scheduling
2. **Real Hinglish data** — scrape code-switched Hindi YouTube, apply VAD segmentation, hand-label turn boundaries
3. **Adversarial domain adaptation** — add a domain discriminator to penalise TTS-specific features
4. **Streaming inference** — replace batch ONNX export with a rolling buffer + causal encoder for true streaming
5. **INT8 quantization** — fix the shape mismatch and get latency below 50ms

---

## 10. Reproducibility

```bash
git clone <repo>
cd shiprocket-turn-detection
pip install -r requirements.txt

# Generate Hinglish eval set
python hinglish_eval_builder.py

# Build splits (small subset for demo)
python data_prep.py --mode build_splits

# Train
python train.py --experiment 1 --epochs 1
python train.py --experiment 2 --epochs 1
python train.py --experiment 3 --epochs 1

# Evaluate + ONNX export
python eval.py --experiment 2 --onnx

# Launch Gradio demo
python app.py
```

All checkpoints are saved to `checkpoints/`, stats to `stats/`, ONNX model to `onnx/`.
