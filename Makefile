.PHONY: setup hinglish analyze splits train1 train2 train3 eval2 eval3 demo clean all

# ── Setup ──────────────────────────────────────────────────────────────────────
setup:
	pip install -r requirements.txt

# ── Data ───────────────────────────────────────────────────────────────────────
hinglish:
	python hinglish_eval_builder.py

analyze:
	python data_prep.py --mode analyze --sample_size 15000

splits:
	python data_prep.py --mode build_splits

# ── Training ───────────────────────────────────────────────────────────────────
train1:
	python train.py --experiment 1 --epochs 1

train2:
	python train.py --experiment 2 --epochs 3

train3:
	python train.py --experiment 3 --epochs 2

# ── Evaluation ─────────────────────────────────────────────────────────────────
eval2:
	python eval.py --experiment 2 --onnx

eval3:
	python eval.py --experiment 3 --onnx

# ── Demo ───────────────────────────────────────────────────────────────────────
demo:
	python app.py

# ── Run everything from scratch ────────────────────────────────────────────────
all: setup hinglish splits train2 eval2
	@echo "Full pipeline done. Run 'make demo' to launch the Gradio app."

# ── Clean generated artefacts (keeps checkpoints and ONNX) ────────────────────
clean:
	rm -rf splits/ cache/ hin_samples/ __pycache__/ stats/plots/*.png
	@echo "Cleaned intermediate files. Checkpoints and ONNX models kept."
