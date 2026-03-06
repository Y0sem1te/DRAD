#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" ./preprocess/preprocess_video_frames.py \
  --json-paths \
    ./data/msr-vtt/msrvtt_train_official.json \
    ./data/msr-vtt/msrvtt_test_official.json \
    ./data/msr-vtt/msrvtt_val_official.json \
  --video-dir ./data/msr-vtt/MSRVTT/videos/all \
  --output-dir ./preprocess/npys \
  --num-frames 12 \
  --overwrite
