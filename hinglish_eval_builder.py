"""
hinglish_eval_builder.py — Supplementary Hinglish Eval Set Builder
====================================================================
Generates 60 synthetic Hinglish audio clips for the held-out eval set:
  - 30 endpoint_bool=True  (complete turns, natural Hinglish code-switching)
  - 30 endpoint_bool=False (mid-turn: fillers, trailing off, pauses)

Synthesis uses gTTS (Google TTS) with language 'hi' for authentic Hindi
phonology. English code-switched words are embedded naturally.

After generation, the script extracts Whisper features for direct
use in eval.py without needing the full data pipeline.

Usage:
    python hinglish_eval_builder.py
"""

import csv
import io
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

BASE_DIR = Path(__file__).parent
HINGLISH_DIR = BASE_DIR / "hinglish_eval"
HINGLISH_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# SCRIPT DEFINITIONS
# Format: (text_for_tts, language_code_for_gtts, description)
# We use 'hi' for Hindi-dominant Hinglish, 'en' for English-dominant
# ──────────────────────────────────────────────────────────────────────────────

# endpoint_bool = True  (genuine turn-ends)
TRUE_SCRIPTS = [
    # Complete sentences, declarative, clear ending
    ("Mera order kal deliver ho jayega", "hi", "order delivery complete"),
    ("Main kal online payment kar doonga", "hi", "payment intent complete"),
    ("Shiprocket se shipment track karo", "hi", "track shipment complete"),
    ("Mujhe refund chahiye please", "hi", "refund request complete"),
    ("Order cancel kar do", "hi", "cancel request complete"),
    ("Address update kar dena", "hi", "address update request"),
    ("Meri complaint register karo", "hi", "complaint complete sentence"),
    ("Package damage ho gaya hai", "hi", "damage report complete"),
    ("Delivery partner ka number do", "hi", "contact request complete"),
    ("Cod available hai is order mein", "hi", "COD query answered"),
    ("Tracking ID share karo mere saath", "hi", "tracking request"),
    ("Return policy kya hai Shiprocket ki", "hi", "policy question complete"),
    ("Mera account verify nahi ho raha", "hi", "verification issue"),
    ("Pincode service nahi hai yahan", "hi", "service unavailability"),
    ("Estimated delivery kab hai", "hi", "ETA query"),
    ("Main complaint file karna chahta hoon", "hi", "complaint intent"),
    ("Invoice download kaise karte hain", "hi", "invoice query"),
    ("International shipping available hai", "hi", "intl shipping query"),
    ("Weight mismatch issue hai mera", "hi", "weight issue complete"),
    ("Bulk order discount milega kya", "hi", "bulk discount query"),
    ("Support team se baat karni hai", "hi", "support escalation"),
    ("Order already shipped ho gaya", "hi", "status update complete"),
    ("Warehouse se pickup kab hoga", "hi", "pickup query"),
    ("RTO charge kyun laga mujhe", "hi", "RTO charge query"),
    ("Mujhe naya label chahiye", "hi", "label request"),
    ("Seller account suspend ho gaya", "hi", "account issue"),
    ("COD amount credit ho gaya account mein", "hi", "COD credit confirmed"),
    ("API integration mein issue aa raha hai", "hi", "API issue complete"),
    ("Dashboard login nahi ho raha", "hi", "login issue complete"),
    ("Main satisfied hoon service se", "hi", "satisfaction complete"),
]

# endpoint_bool = False  (mid-turn: fillers, hesitation, trailing off)
FALSE_SCRIPTS = [
    # Trailing fillers — sentence not complete, sounds like more is coming
    ("Matlab woh jo order tha na", "hi", "trailing 'na' filler"),
    ("Toh basically acha", "hi", "toh/acha filler only"),
    ("Haan so ek second", "hi", "haan-so filler pause"),
    ("Woh kya hai na actually", "hi", "trailing 'na' with actually"),
    ("Matlab delivery wali baat kar raha tha", "hi", "mid-thought delivery"),
    ("Toh main bol raha tha ki", "hi", "toh main trailing ki"),
    ("Acha toh", "hi", "short acha-toh filler"),
    ("Haan matlab", "hi", "haan matlab filler"),
    ("Woh ek order tha na mera", "hi", "tha-na trailing"),
    ("Bas woh issue jo hai", "hi", "bas woh trailing"),
    ("Basically acha actually dekho", "hi", "multiple fillers"),
    ("Matlab woh issue basically", "hi", "matlab basically"),
    ("Toh tracking wala page", "hi", "incomplete trailing"),
    ("Acha woh delivery partner", "hi", "acha incomplete"),
    ("Haan toh woh jo tha na", "hi", "haan toh tha na"),
    ("Main soch raha tha ki", "hi", "soch raha trailing ki"),
    ("Woh actually matlab", "hi", "woh actually matlab"),
    ("Toh acha bhai woh order", "hi", "toh acha bhai"),
    ("Matlab basically acha", "hi", "multiple filler chain"),
    ("Haan woh package wala issue na", "hi", "woh issue na"),
    ("So basically delivery ka issue", "hi", "so basically trailing"),
    ("Acha toh woh jo", "hi", "acha toh woh incomplete"),
    ("Haan haan acha", "hi", "haan haan filler"),
    ("Woh ek second actually", "hi", "trailing actually"),
    ("Matlab", "hi", "single matlab filler"),
    ("Toh", "hi", "single toh filler"),
    ("Acha acha", "hi", "acha repeated"),
    ("Bas ek second woh issue hai na", "hi", "bas ek second hai na"),
    ("Woh matlab jo order hai", "hi", "woh matlab incomplete"),
    ("Actually woh dekho na", "hi", "actually trailing na"),
]

assert len(TRUE_SCRIPTS) == 30, f"Expected 30 true scripts, got {len(TRUE_SCRIPTS)}"
assert len(FALSE_SCRIPTS) == 30, f"Expected 30 false scripts, got {len(FALSE_SCRIPTS)}"


def synthesize_clip(text: str, lang: str, out_path: Path, target_sr: int = 16000) -> bool:
    """Synthesize TTS audio using gTTS, save as 16kHz mono WAV."""
    try:
        from gtts import gTTS
        from pydub import AudioSegment

        tts = gTTS(text=text, lang=lang, slow=False)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name
            tts.save(tmp_path)

        # Convert mp3 → wav 16kHz mono
        audio = AudioSegment.from_mp3(tmp_path)
        audio = audio.set_frame_rate(target_sr).set_channels(1)
        audio.export(str(out_path), format="wav")
        os.unlink(tmp_path)
        return True
    except Exception as e:
        print(f"    ⚠ TTS failed for '{text[:30]}...': {e}")
        # Fallback: generate 1s of silence-ish noise as placeholder
        sr = target_sr
        arr = np.random.randn(sr).astype(np.float32) * 0.001
        sf.write(str(out_path), arr, sr)
        return False


def extract_whisper_features(wav_path: Path, tail_sec: float = 2.0, target_sr: int = 16000) -> np.ndarray:
    """Extract Whisper log-Mel features from a WAV file (tail window)."""
    from transformers import WhisperFeatureExtractor
    import librosa

    extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")
    array, sr = sf.read(str(wav_path), dtype="float32")
    if array.ndim > 1:
        array = array.mean(axis=1)
    if sr != target_sr:
        array = librosa.resample(array, orig_sr=sr, target_sr=target_sr)

    # Tail window
    tail_samples = int(tail_sec * target_sr)
    if len(array) > tail_samples:
        array = array[-tail_samples:]
    else:
        pad = tail_samples - len(array)
        array = np.pad(array, (pad, 0), mode="constant")

    feats = extractor(array, sampling_rate=target_sr, return_tensors="np")
    return feats.input_features[0]  # (80, 3000)


def build_hinglish_eval_set():
    print("="*65)
    print("BUILDING SUPPLEMENTARY HINGLISH EVAL SET")
    print("="*65)
    print(f"Output: {HINGLISH_DIR}/")
    print(f"  30 True (turn-end) + 30 False (mid-turn) clips")
    print(f"  Synthesis: gTTS with Hindi phonology + code-switched text\n")

    all_scripts = (
        [(text, lang, desc, 1) for text, lang, desc in TRUE_SCRIPTS] +
        [(text, lang, desc, 0) for text, lang, desc in FALSE_SCRIPTS]
    )

    metadata = []
    features_list = []
    success_count = 0

    for i, (text, lang, desc, label) in enumerate(tqdm(all_scripts, desc="Synthesizing")):
        label_str = "true" if label == 1 else "false"
        wav_path = HINGLISH_DIR / f"hing_{i:03d}_{label_str}.wav"

        # TTS synthesis
        ok = synthesize_clip(text, lang, wav_path)
        if ok:
            success_count += 1
        time.sleep(0.5)  # Rate limit for gTTS

        # Feature extraction
        try:
            feats = extract_whisper_features(wav_path)
        except Exception as e:
            print(f"  ⚠ Feature extraction failed: {e}, using zeros")
            feats = np.zeros((80, 3000), dtype=np.float32)

        features_list.append(feats)
        metadata.append({
            "idx": i,
            "filename": wav_path.name,
            "text": text,
            "language": lang,
            "description": desc,
            "label": label,
            "endpoint_bool": bool(label),
            "synthetic": True,
        })

    # Save features
    feats_arr = np.array(features_list, dtype=np.float32)
    np.save(str(HINGLISH_DIR / "features.npy"), feats_arr)
    print(f"\n✓ Features saved: {feats_arr.shape}")

    # Save metadata CSV
    meta_path = HINGLISH_DIR / "metadata.csv"
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=metadata[0].keys())
        writer.writeheader()
        writer.writerows(metadata)
    print(f"✓ Metadata saved: {meta_path}")

    # Save metadata JSON too
    with open(HINGLISH_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Summary
    df_meta = {r["label"]: 0 for r in metadata}
    for r in metadata:
        df_meta[r["label"]] = df_meta.get(r["label"], 0) + 1

    print(f"\n  TTS success rate: {success_count}/{len(all_scripts)}")
    print(f"  Label distribution: {df_meta}")
    print(f"  Files written: {len(all_scripts)} WAVs + features.npy + metadata.csv")
    print(f"\n  ✓ Hinglish eval set ready at {HINGLISH_DIR}/")
    print("    Use in eval.py: python eval.py --experiment 2")


if __name__ == "__main__":
    build_hinglish_eval_set()
