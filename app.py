"""
app.py — Gradio Demo: Hinglish Turn Detection
===============================================
Upload or record audio → get turn-end/mid-turn prediction with confidence.

Deploy to HF Space:
    gradio deploy

Local run:
    python app.py
"""

import os
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf

BASE_DIR = Path(__file__).parent
ONNX_DIR = BASE_DIR / "onnx"
HINGLISH_DIR = BASE_DIR / "hinglish_eval"

# ── Load model at startup ──────────────────────────────────────────────────────
def load_inference_session():
    """Load the best available ONNX model (exp2 preferred)."""
    import onnxruntime as ort
    for exp in [2, 3, 1]:
        path = ONNX_DIR / f"exp{exp}_model.onnx"
        if path.exists():
            print(f"Loading ONNX model: {path}")
            sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            return sess, exp
    return None, None

SESSION, LOADED_EXP = load_inference_session()


def preprocess_audio(audio_input, target_sr: int = 16000, tail_sec: float = 2.0) -> np.ndarray:
    """
    Accept either (sr, array) tuple from Gradio mic or a file path.
    Returns WhisperFeatureExtractor features: (1, 80, 3000)
    """
    from transformers import WhisperFeatureExtractor
    import librosa

    extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")

    if isinstance(audio_input, tuple):
        sr, array = audio_input
        if array.ndim > 1:
            array = array.mean(axis=1)
        array = array.astype(np.float32)
        if np.abs(array).max() > 1.0:
            array = array / 32768.0  # int16 → float32
    elif isinstance(audio_input, str):
        array, sr = sf.read(audio_input, dtype="float32")
        if array.ndim > 1:
            array = array.mean(axis=1)
    else:
        raise ValueError(f"Unknown audio input type: {type(audio_input)}")

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
    return feats.input_features  # (1, 80, 3000)


def predict(audio_input):
    """Main inference function called by Gradio."""
    if audio_input is None:
        return (
            "⚠️ No audio provided",
            None,
            "Please record or upload an audio clip",
        )

    if SESSION is None:
        return (
            "⚠️ Model not loaded",
            None,
            "ONNX model not found. Run: python train.py --experiment 2 && python eval.py --experiment 2 --onnx",
        )

    try:
        features = preprocess_audio(audio_input)
        logits = SESSION.run(None, {"input_features": features.astype(np.float32)})[0]
        prob = float(1 / (1 + np.exp(-logits[0, 0])))  # sigmoid

        is_turn_end = prob >= 0.5
        confidence = prob if is_turn_end else (1 - prob)

        if is_turn_end:
            label = f"✅ Turn Complete  (confidence: {confidence*100:.1f}%)"
            explanation = (
                f"**Prediction:** The user has finished their turn.\n\n"
                f"**Confidence:** {confidence*100:.1f}%  |  **Raw score:** {prob:.4f}\n\n"
                f"The model detected prosodic and acoustic cues indicating a genuine turn-end — "
                f"falling intonation, complete phrasing, absence of trailing fillers."
            )
        else:
            label = f"🔄 Still Speaking… (confidence: {confidence*100:.1f}%)"
            explanation = (
                f"**Prediction:** The user is mid-turn (pausing, filler word, or trailing off).\n\n"
                f"**Confidence:** {confidence*100:.1f}%  |  **Raw score:** {prob:.4f}\n\n"
                f"The model detected cues of an incomplete turn — possibly a filler word like "
                f"'matlab', 'toh', 'haan', a breath pause, or rising intonation suggesting "
                f"more speech to come."
            )

        # Confidence bar as HTML
        bar_color = "#2ecc71" if is_turn_end else "#e74c3c"
        bar_html = f"""
        <div style="background:#1e1e2e; border-radius:12px; padding:20px; font-family:Inter,sans-serif;">
          <div style="font-size:22px; font-weight:700; color:{'#2ecc71' if is_turn_end else '#e74c3c'}; margin-bottom:12px;">
            {label}
          </div>
          <div style="background:#2a2a3e; border-radius:8px; height:20px; overflow:hidden; margin-bottom:8px;">
            <div style="height:100%; width:{confidence*100:.1f}%; background:{bar_color};
                        border-radius:8px; transition:width 0.5s ease;"></div>
          </div>
          <div style="color:#888; font-size:13px;">Confidence: {confidence*100:.1f}%  |  Score: {prob:.4f}</div>
          <div style="margin-top:14px; color:#ccc; font-size:14px; line-height:1.6;">
            Model: Whisper Tiny Encoder + Attention Pooling + MLP Head<br>
            Experiment: {LOADED_EXP} | Input: last 2s tail window
          </div>
        </div>
        """
        return bar_html, None, explanation

    except Exception as e:
        return f"❌ Error: {str(e)}", None, f"```\n{str(e)}\n```"


# ── Example clips ─────────────────────────────────────────────────────────────
def get_example_clips():
    """Return a few example WAV paths from the Hinglish eval set."""
    examples = []
    if HINGLISH_DIR.exists():
        true_clips = sorted(HINGLISH_DIR.glob("*_true.wav"))[:2]
        false_clips = sorted(HINGLISH_DIR.glob("*_false.wav"))[:2]
        examples = [str(p) for p in true_clips + false_clips]
    return examples


# ── Gradio UI ─────────────────────────────────────────────────────────────────
CSS = """
body { background: #0f0f1a; font-family: 'Inter', sans-serif; }
.gradio-container { background: #0f0f1a !important; }
.panel { background: #1a1a2e !important; border: 1px solid #2a2a4e !important; border-radius: 16px !important; }
h1 { background: linear-gradient(135deg, #667eea, #764ba2); -webkit-background-clip: text;
     -webkit-text-fill-color: transparent; font-size: 2.2em !important; font-weight: 800 !important; }
.subtitle { color: #8888aa; font-size: 0.95em; margin-bottom: 20px; }
"""

with gr.Blocks(css=CSS, title="Hinglish Turn Detection", theme=gr.themes.Base()) as demo:
    gr.Markdown("""
# 🎤 Hinglish Turn Detection
### Shiprocket Voice AI — Real-time End-of-Turn Detector

Built on **Whisper Tiny encoder** + **Attention Pooling** + **MLP Head**.
Predicts from raw audio whether a user has finished speaking or is mid-thought.
Optimised for **Indian Hinglish** (code-switched Hindi/English) with filler words like
*matlab*, *toh*, *haan*, *acha*, *bas*.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="🎙️ Record or Upload Audio",
            )
            submit_btn = gr.Button("🔍 Detect Turn", variant="primary", size="lg")

            gr.Markdown("**Try these Hinglish examples:**")
            example_clips = get_example_clips()
            if example_clips:
                examples = gr.Examples(
                    examples=[[clip] for clip in example_clips],
                    inputs=audio_input,
                    label="Example clips (from Hinglish eval set)",
                )

        with gr.Column(scale=1):
            result_html = gr.HTML(label="Prediction")
            explanation = gr.Markdown(label="Explanation")

    submit_btn.click(
        fn=predict,
        inputs=[audio_input],
        outputs=[result_html, gr.Audio(visible=False), explanation],
    )

    gr.Markdown("""
---
### About this model
| Property | Value |
|----------|-------|
| Backbone | Whisper Tiny (encoder-only, 39M params) |
| Pooling | Attention pooling (learned query) |
| Head | 2-layer MLP (384→128→1) |
| Input | Last 2s of audio clip, 80-channel log-Mel |
| Trainable params | ~2.5M (remaining params frozen) |
| Dataset | pipecat-ai/smart-turn-data-v3.2-train (271k rows) |
| Target domain | Indian Hinglish (code-switched Hindi/English) |

*Built for the Shiprocket Data Scientist ML hiring challenge.*
    """)


if __name__ == "__main__":
    demo.launch(share=True)
