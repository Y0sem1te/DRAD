#!/bin/bash

# bash run_eval_multiple_lora.sh

echo "=========================================="
echo "beginning evaluation of multiple LoRA weights"
echo "=========================================="

EVAL_SCRIPT="./eval_metrics_rag.py"

LORA_DIRS="
./train_output/20260125154422/epoch_15
./train_output/20260125234058/epoch_15
./train_output/20260126090738/epoch_7
"

# TRAIN_OUTPUT_ROOT="./train_output/20260123000734"
# LORA_DIRS=$(find $TRAIN_OUTPUT_ROOT -maxdepth 1 -type d -name "epoch_*" | sort -V)

TOP_K=5
MAX_LOOKBACK=8
SIMILARITY_THRESHOLD=0.85

for lora_dir in $LORA_DIRS; do
    if [ -z "$lora_dir" ]; then
        continue
    fi
    
    echo ""
    echo "=========================================="
    echo "begin evaluation LoRA: $lora_dir"
    echo "=========================================="
    echo ""
    
    if [ ! -d "$lora_dir" ]; then
        echo "⚠️ alert: don't exist: $lora_dir"
        continue
    fi
    
    python $EVAL_SCRIPT \
        --lora_dir="$lora_dir" \
        --top_k=$TOP_K \
        --max_lookback=$MAX_LOOKBACK \
        --similarity_threshold=$SIMILARITY_THRESHOLD
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ evaluation done: $(basename $lora_dir)"
    else
        echo ""
        echo "❌ evaluation false: $(basename $lora_dir), exit code: $?"
        echo "continue to evaluate the next lora..."
    fi
    
    echo ""
done

echo ""
echo "=========================================="
echo "all evaluation done"
echo "=========================================="
echo ""
echo "Evaluated the following LoRA weights:"
for lora_dir in $LORA_DIRS; do
    if [ -n "$lora_dir" ] && [ -d "$lora_dir" ]; then
        echo "  - $(basename $lora_dir)"
    fi
done
echo ""
echo "You can check the evaluation results in the corresponding LoRA directories."
