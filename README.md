# Hinglish Turn Detection

Audio-only turn endpoint detection — Shiprocket Data Scientist Challenge

A compact, fast model that listens to the trailing two seconds of a speaker's audio and answers one question: has the speaker finished their turn, or are they pausing mid-thought?

---

## Headline Results

| Experiment | Configuration | Val AUROC | Test AUROC | Test F1 | Hinglish AUROC | CPU Latency |
|------------|---------------|-----------|------------|---------|----------------|-------------|
| Exp 1 | Baseline — frozen Whisper + mean pool | 0.464 | — | — | — | — |
| Exp 2 | + Attention pooling, top-2 blocks unfrozen | 0.930 | **0.861** | **0.791** | **0.782** | **207 ms / RTF 0.103** |
| Exp 3 | + Hard-negative 3x oversampling | **0.901** | 0.911 | 0.863 | 0.720 | 508 ms / RTF 0.254 |

Experiment 2 is selected as the production checkpoint: it achieves the best Hinglish generalisation while running at one-tenth real-time on a single CPU thread.

---

## Architecture

```
Waveform (16 kHz PCM)
  -> WhisperFeatureExtractor  [frozen]  -> log-mel spectrogram (80-dim, 3000 frames)
  -> Whisper Tiny Encoder
       Blocks 0-1: frozen    (low-level acoustic features)
       Blocks 2-3: fine-tuned (high-level temporal patterns)
  -> AttentionPooling         [learned 384-dim query, softmax over time axis]
  -> MLP Head                 Linear(384->128) -> GELU -> Dropout(0.15) -> Linear(128->1)
  -> sigmoid -> P(turn_end)
```

- 8.26M total parameters; only 50K trainable in Exp 2 (0.61% of backbone)
- Exported to ONNX (36 KB); runs at RTF 0.103 on a single CPU thread
- No ASR, no transcript — pure acoustic signal

---

## The Hinglish Problem (Honest Reporting)

The training dataset (`pipecat-ai/smart-turn-data-v3.2-train`) tags 721 rows as `hin`, but inspection confirms these are 100% synthetic TTS clips from a single voice model (`chirp3_1`). No code-switched Hinglish is present.

To surface this gap explicitly, we constructed a 60-clip synthetic Hinglish evaluation set (30 turn-end, 30 mid-turn) using gTTS with code-switched Hindi+English sentences.

| Evaluation Set | AUROC | F1 | Note |
|----------------|-------|-----|------|
| Main test set (in-distribution) | 0.861 | 0.791 | Matched training distribution |
| Hinglish held-out (out-of-distribution) | 0.782 | 0.789 | Synthetic proxy, not gold standard |
| Gap | -0.079 | -0.003 | Honest reporting of distribution shift |

The AUROC drops roughly 8 points on out-of-distribution Hinglish. This is reported as-is rather than obscured.

---

## Key Findings

1. **Attention pooling is the decisive change.** Exp 1 to Exp 2 moves AUROC from 0.464 to 0.930 on validation without changing a single encoder weight — only the pooling strategy changes. Concentrating on prosodically salient frames near the turn boundary matters far more than the depth of fine-tuning.

2. **The model partially learns TTS voice characteristics.** The `midcentury_1` source (real human speech) scores AUROC 0.460 — near random. The model is not production-ready without more real-speech training data.

3. **Hard-negative oversampling helps on mid-utterance fillers.** Exp 3 (3x oversampling of `midfiller=True` negatives) raises test AUROC to 0.911, but Hinglish AUROC drops to 0.720 — the model becomes more aggressive at predicting non-endpoints, which hurts on the Hinglish set where the decision boundary differs.

4. **ONNX at 207 ms, RTF 0.103.** The exported model comfortably fits within the latency budget of a voice assistant turn-taking decision.

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Build the synthetic Hinglish evaluation set (gTTS, ~5 min, requires internet)
python hinglish_eval_builder.py

# 3. Analyse the dataset
python data_prep.py --mode analyze --sample_size 15000

# 4. Build train / val / test splits
python data_prep.py --mode build_splits

# 5. Train (three experiments)
python train.py --experiment 1 --epochs 1
python train.py --experiment 2 --epochs 3
python train.py --experiment 3 --epochs 2

# 6. Evaluate with ONNX export and latency benchmark
python eval.py --experiment 2 --onnx
python eval.py --experiment 3 --onnx

# 7. Launch the Gradio demo
python app.py
```

---

## Repository Structure

```
hinglish-turn-detector/
|
|- model.py                   # WhisperTinyTurnDetector + AttentionPooling
|- data_prep.py               # EDA analysis and split builder
|- train.py                   # Training loop (Experiments 1, 2, 3)
|- eval.py                    # Test evaluation + Hinglish eval + ONNX benchmark
|- hinglish_eval_builder.py   # Synthetic Hinglish eval set via gTTS
|- app.py                     # Gradio interactive demo
|- requirements.txt
|- REPORT.md                  # Full analysis and lab report
|
|- stats/
|   |- data_analysis.json         # EDA statistics (15,000-row sample)
|   |- exp1_results.json          # Experiment 1 training summary
|   |- exp2_results.json          # Experiment 2 training summary
|   |- exp3_results.json          # Experiment 3 training summary
|   |- exp2_full_eval.json        # Experiment 2 full evaluation (test + Hinglish + latency)
|   |- exp3_full_eval.json        # Experiment 3 full evaluation
|   |- exp2_errors.json           # Top-15 misclassified samples, Exp 2
|   |- exp3_errors.json           # Top-15 misclassified samples, Exp 3
|   |- plots/
|       |- eda_summary.png        # EDA overview (6-panel figure)
|       |- exp1_curves.png        # Exp 1 training curves
|       |- exp2_curves.png        # Exp 2 training curves
|       |- exp3_curves.png        # Exp 3 training curves
|       |- exp2_test_cm.png       # Exp 2 confusion matrix (main test set)
|       |- exp2_hinglish_cm.png   # Exp 2 confusion matrix (Hinglish set)
|       |- exp3_test_cm.png       # Exp 3 confusion matrix (main test set)
|       |- exp3_hinglish_cm.png   # Exp 3 confusion matrix (Hinglish set)
|
|- onnx/
|   |- exp2_model.onnx            # ONNX export, Experiment 2 (best checkpoint)
|   |- exp3_model.onnx            # ONNX export, Experiment 3
|
|- checkpoints/                   # Best .pt checkpoints (gitignored, large)
|- splits/                        # Preprocessed numpy splits (gitignored, large)
|- hinglish_eval/                 # Hinglish WAV files + metadata (WAVs gitignored)
```

---

## Reproducibility Notes

All random seeds are set to 42. Training was done on CPU with a 2,000-row subset for Exp 1 and Exp 2, and a 2,822-row hard-negative-oversampled subset for Exp 3. Full-scale training on a GPU with the complete 220,000-row dataset is expected to produce substantially better numbers.

See [REPORT.md](REPORT.md) for the full analysis.
