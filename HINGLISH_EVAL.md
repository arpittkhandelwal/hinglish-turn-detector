# Hinglish Evaluation Set — Construction and Limitations

This document explains how the 60-clip Hinglish held-out evaluation set was built, what it tests, and what it does not test.

---

## Why a separate eval set?

The training dataset (`pipecat-ai/smart-turn-data-v3.2-train`) contains 721 rows tagged as `hin`, but all of them are synthetic TTS clips from the `chirp3_1` voice model — monolingual Hindi read aloud by a single synthetic voice. There is no code-switched Hindi+English (Hinglish) anywhere in the training data.

Since the stated goal is robustness on Hinglish — the natural spoken register of most Indian voice assistant users — we needed an evaluation set that actually reflects this use case. Rather than claim the model works on Hinglish without evidence, we built a dedicated set and report the gap honestly.

---

## How it was built

The set was created using `hinglish_eval_builder.py` via the `gTTS` library.

**Turn-end clips (label = 1, 30 clips):**

Complete Hinglish sentences expressing common Shiprocket customer intents — order tracking, delivery queries, refund requests, payment confirmation, etc. Each is a syntactically complete sentence that a real user would say before expecting a response.

Examples:
- "Mera order kal deliver ho jayega" (My order will be delivered tomorrow)
- "Mujhe refund chahiye please" (I need a refund please)
- "API integration mein issue aa raha hai" (There is an issue in API integration)

**Mid-turn clips (label = 0, 30 clips):**

Incomplete utterances ending in common Hinglish filler markers — the kinds of sounds or words a speaker produces mid-thought. These are the hardest cases for any turn detector because they often have prosodic pause signatures without being genuine turn-ends.

Examples:
- "Matlab woh jo order tha na" (Basically that order which was, you know)
- "Toh main bol raha tha ki" (So I was saying that...)
- "Haan matlab" (Yeah like...)
- "Matlab" (a single filler word — hardest possible case)

The full list of sentences with labels is in `hinglish_eval/metadata.csv`.

---

## Limitations

This is a proxy evaluation set, not a gold standard. Specific caveats:

**It is synthetic, not real speech.** gTTS produces a clean, consistent voice without the hesitations, breath sounds, and prosodic irregularities of real human Hinglish conversation. A real evaluation would use recordings from actual Hinglish speakers.

**The sentences are short and unambiguous.** Real conversational Hinglish is longer, messier, and harder to label. The sentences here are deliberately simple to establish a clean signal.

**gTTS Hindi may not capture code-switching prosody.** When a speaker switches between Hindi and English mid-sentence, their prosody shifts in ways that gTTS does not model. The model is evaluated on a simplified version of the actual distribution shift.

**60 clips is a small sample.** With only 30 examples per class, confidence intervals on any metric are wide. The AUROC numbers (0.782 for Exp 2, 0.720 for Exp 3) should be interpreted as directional, not precise.

---

## What the results tell us

The Exp 2 model achieves AUROC 0.782 on this set versus 0.861 on the main test set — a gap of 0.079. This gap is real and expected. The model has seen no genuine Hinglish in training, and the acoustic properties of code-switched speech (mixing prosodic patterns from two languages, Hinglish-specific filler words like *matlab*, *toh*, *haan*, *acha*, *bas*) are genuinely different from what the model was trained on.

The F1 gap is much smaller (0.789 vs 0.791), which means the model is still making roughly the right classification decisions on this set — it is the confidence calibration that degrades, not the direction of the predictions.

The Exp 3 model performs worse on this set (AUROC 0.720, F1 0.709) despite better main test performance. Hard-negative oversampling makes the model more conservative, which hurts on the Hinglish filler cases where the model needs to fire on subtle cues.

---

## Reproducing the eval set

```bash
python hinglish_eval_builder.py
```

This creates `hinglish_eval/*.wav` (60 WAV files at 22050 Hz) and `hinglish_eval/metadata.csv`. The WAV files are gitignored due to size — only the metadata is tracked. Re-running the builder will regenerate identical clips (gTTS output is deterministic for the same text and language).

Once the eval set is built, running:

```bash
python eval.py --experiment 2 --onnx
```

will automatically evaluate on this set and report the gap vs the main test set.
