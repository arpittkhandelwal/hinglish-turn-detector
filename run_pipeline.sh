#!/bin/bash
set -e
source venv/bin/activate
echo "=== Step 1: Build splits (5k/1k/1k) ===" && python data_prep.py --mode build_splits
echo "=== Step 2: Train Exp 2 (3 epochs) ===" && python train.py --experiment 2 --epochs 3
echo "=== Step 3: Train Exp 3 (3 epochs) ===" && python train.py --experiment 3 --epochs 3
echo "=== Step 4: Evaluate best model ===" && python eval.py --experiment 3 --onnx
echo "Pipeline done."
