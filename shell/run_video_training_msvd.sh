#!/bin/bash

export CUDA_VISIBLE_DEVICES="0,1"

TORCH_DISTRIBUTED_DEBUG=DETAIL \
ACCELERATE_DEBUG_VERBOSITY="info" \
accelerate launch \
  --main_process_port=29920 \
  --mixed_precision=bf16 \
  --dynamo_backend=no \
  --num_machines=1 \
  --num_processes=2 \
  --use_deepspeed \
  finetune_distributed_video.py \
  --use_lora True \
  --seed 42 \
  --train-json ./data/MSVD/msvd_train.json \
  --eval-json ./data/MSVD/msvd_val.json \
  --test-json ./data/MSVD/msvd_test.json \
  --video-dir ./data/MSVD/raw_videos/ \
  --learning-rate 1e-6 \
  --num-frames 12 \
  --max-history 4 \
  --use-teacher-forcing \
  --teacher-forcing-k 700