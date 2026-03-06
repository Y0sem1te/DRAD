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
  finetune_distributed_video_rapid.py \
  --use_lora True \
  --seed 42 \
  --train-json ./data/msr-vtt/msrvtt_train_official.json \
  --eval-json ./data/msr-vtt/msrvtt_val_official.json \
  --test-json ./data/msr-vtt/msrvtt_test_official.json \
  --video-dir ./data/msr-vtt/MSRVTT/videos/all/ \
  --frame-cache-dir ./preprocess/npys \
  --learning-rate 1e-6 \
  --num-frames 12 \
  --max-history 4 \
  --use-teacher-forcing \
  --teacher-forcing-k 3025