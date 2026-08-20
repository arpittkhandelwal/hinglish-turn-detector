# Hinglish Turn Detection — Report

**Task:** Audio-only turn endpoint detection for Hinglish voice interfaces
**Dataset:** pipecat-ai/smart-turn-data-v3.2-train (HuggingFace)

---

## What I Was Trying to Solve

The problem is simple to state but tricky to solve: given the last couple seconds of someone speaking, can a model figure out whether they have finished their thought or are just pausing mid-sentence? This matters a lot for voice assistants — if the system jumps in too early, it interrupts the user; if it waits too long, the conversation feels sluggish.

The extra layer of complexity here is Hinglish — the way most people in India actually talk, mixing Hindi and English in the same sentence. The training data does not really have this, so I had to be honest about what the model can and cannot do on real Hinglish speech.

---

## Looking at the Data First

Before writing a single line of model code, I sampled 15,000 rows from the dataset to understand what I was working with.

The class balance turned out to be almost perfect — 49.6% turn-ends and 50.4% mid-turns. That was a relief because it meant I did not have to do any loss reweighting for the baseline.

What surprised me more was the hard negatives. Almost 43% of all the negative examples (mid-turns) had `midfiller=True` — these are clips where the speaker is doing the "umm" and "uh" thing, or trailing off with a hesitation sound. These are genuinely hard because acoustically they can look a lot like a turn-end. I flagged these early and came back to them in Experiment 3.

The other thing I noticed right away: the `spoken_text` field was empty for every single row in my sample. So the model had to be purely acoustic — no text shortcut available even if I had wanted one.

The Hinglish situation was something I had to dig into separately. The training set has 721 rows tagged as `hin`, but when you look at them, they are all synthetic TTS clips from the same voice model (`chirp3_1`). There is no actual code-switched Hindi+English speech in there. I thought about just pretending this was fine, but it felt dishonest, so I built a small synthetic Hinglish evaluation set myself instead (more on that below).

---

## The Model

I went with Whisper Tiny as the backbone. A few reasons for this:

- Whisper was trained on a lot of multilingual speech, including Indic languages, so the representations should transfer better to Hindi than an English-only model would
- I only needed the encoder — the decoder is for transcription, and I do not need transcription, I just need acoustic features
- It is small enough to run on CPU at real-time, which was a hard requirement

The architecture ended up being:

The audio goes through WhisperFeatureExtractor to get an 80-dimensional log-mel spectrogram. That feeds into the Whisper Tiny encoder (4 transformer blocks). I tried freezing different amounts of it across the three experiments. The encoder output is a sequence of frame-level vectors, which then get pooled down to a single vector, and a small two-layer MLP on top outputs a single logit. Sigmoid gives the probability.

The pooling step turned out to be the most important design decision of the whole project.

---

## Three Experiments

I ran three experiments, each building on the last one.

### Experiment 1 — Does off-the-shelf Whisper even have signal for this?

I froze the entire Whisper encoder and just trained a small MLP head on top of mean-pooled features. If this worked, it would mean Whisper's default representations already encode turn-end information and I could get away with a very cheap model.

It did not work. Val AUROC came out at 0.464 — that is basically a coin flip. Mean pooling over all frames washes out whatever signal exists at the end of the clip. The model is averaging over the full 2 seconds equally, including the silence at the beginning and the middle of the clip, and the end-of-turn prosodic cues get diluted.

### Experiment 2 — Attention pooling makes everything better

Instead of averaging all frames equally, I added a learned attention query that figures out which frames to pay attention to. The idea is that the model should naturally learn to focus on the last fraction of a second — where intonation, energy, and timing are most informative for turn-end detection.

This was a big jump. Val AUROC went from 0.464 to 0.930 just by changing the pooling. I also unfroze the top two encoder blocks so they could fine-tune to the task, which helps but the pooling is clearly the dominant factor.

Best checkpoint: epoch 3, val AUROC 0.930, val F1 0.809.

### Experiment 3 — Dealing with the hard negatives

The midfiller cases kept showing up in my error analysis. In Experiment 3 I oversampled all the hard negatives (midfiller=True AND endpoint_bool=False) by 3x in the training split, hoping the model would learn to not be fooled by hesitation sounds.

Val AUROC went to 0.901. On the main test set the numbers looked better across the board. But when I ran it on the Hinglish set, performance dropped compared to Experiment 2. The model had become more conservative about predicting turn-ends, which hurt on the Hinglish clips where the pattern is different from what it was trained on.

So Experiment 2 is the one I am going with as the recommended checkpoint.

---

## Results

### Main test set (500 clips)

Experiment 2 hit 0.786 accuracy, 0.791 F1, and 0.861 AUROC. Experiment 3 was better on the main set (0.850 accuracy, 0.863 F1, 0.911 AUROC) but worse on Hinglish, so the trade-off did not feel worth it.

Confusion matrix for Experiment 2:

```
                  Predicted False   Predicted True
Actual False            190               56
Actual True              51              203
```

The model is slightly more likely to call something a turn-end than to miss one. I think that is the right bias for a voice assistant — better to occasionally interrupt slightly early than to sit there waiting.

### Per language

The `hin` row actually scores the highest of any language (AUROC 0.936, F1 0.867). This sounds great but it is misleading — the Hindi test clips are from the same synthetic TTS voice as the training data, so the model has essentially memorised that voice's acoustic signature. This is not Hinglish generalisation, it is TTS memorisation.

English surprisingly scores the lowest despite being the most common language. I think what is happening is that the test set has more real human English speech (from `liva_1` and `midcentury_1`) than the training set does, and the model struggles with real speech.

### The Hinglish set

I built 60 synthetic clips using gTTS with code-switched sentences — half turn-ends, half mid-turns. It is not a gold standard (it is still synthetic), but it gives an honest out-of-distribution signal.

Experiment 2 on the Hinglish set: AUROC 0.782, F1 0.789. That is a drop of 0.079 AUROC points and 0.003 F1 points from the main test set. The F1 gap is tiny, which is reassuring, but the AUROC drop suggests the model is less well-calibrated on code-switched speech — it is still making roughly the right decisions but with less confident, more spread-out probability scores.

Experiment 3 on Hinglish was worse: AUROC 0.720, F1 0.709. The hard-negative oversampling made things worse here, not better.

### Source breakdown — the uncomfortable finding

| Source | AUROC (Exp 2) |
|--------|---------------|
| chirp3_1 (synthetic TTS) | 0.905 |
| chirp3_2 (synthetic TTS) | 0.876 |
| liva_1 (real human speech) | 0.789 |
| midcentury_1 (real human speech) | 0.460 |

That last number, 0.460, is essentially random. The model is nearly useless on `midcentury_1`, which is real conversational human speech. This is the most important failure mode in the whole project. The model has learned to recognise the prosodic style of the synthetic TTS systems in the training data, not the language-universal features of turn-ending. It will struggle in production until this is addressed.

---

## Error Analysis

I looked at the 15 most confidently wrong predictions for each experiment.

For Experiment 2, the pattern was consistent: false negatives (missed turn-ends) almost all come from `liva_1` and `midcentury_1` — real human speech. False positives (wrong turn-end calls) often involve `midfiller=True` cases where the model gets fooled by hesitation prosody.

For Experiment 3, the errors shifted. After hard-negative oversampling the model became more conservative, and the false negatives shifted toward less common languages — Vietnamese, Polish, Chinese — where it seems the model is now under-triggering. The false positives are also more confident (probability > 0.93) which is a calibration problem.

---

## Latency

I exported both Experiment 2 and Experiment 3 to ONNX and benchmarked on CPU with a single thread.

| | Experiment 2 | Experiment 3 |
|-|-------------|-------------|
| Model size | 0.036 MB | 0.036 MB |
| Mean latency | 207 ms | 508 ms |
| P95 latency | 209 ms | 679 ms |
| Real-time factor | 0.103 | 0.254 |

Experiment 2 runs in about 207ms to process a 2-second clip — that is 10% of real-time, comfortably fast enough for a live voice assistant. The 36 KB file size is tiny.

The ONNX output matches PyTorch output to within 2e-7 for both models, so the export is numerically faithful.

I tried INT8 quantisation but hit a shape inference error in the AttentionPooling layer under dynamic batch axes. Skipped it for now — 207ms is already well within budget without it.

---

## What I Would Do Next

A few things I would tackle with more time:

Training on the full dataset is the obvious first step. I ran everything on CPU with about 2,000 rows per experiment due to time constraints. Running on a GPU with all 220,000 rows for 5-10 epochs would close most of the gaps I am seeing.

The real-speech problem is the one I am most concerned about. I would want to collect or find more data from the `liva_1`-style sources — real conversational human speech, not TTS — and make sure that is well represented in training. The `midcentury_1` AUROC of 0.46 is the thing that would keep me up at night if this were going to production.

For Hinglish specifically: collect real code-switched speech. Even a few hundred labelled clips from YouTube Hindi-English conversations would let me fine-tune on something real instead of gTTS proxies.

Streaming inference would be the next architectural change. Right now the model takes a fixed 2-second batch. A production voice assistant needs a rolling buffer with a causal encoder, deciding in real-time as speech comes in rather than waiting for a chunk to finish.
