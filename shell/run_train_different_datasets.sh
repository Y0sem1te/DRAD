#!/bin/bash

# bash run_train_different_datasets.sh

echo "=========================================="
echo "Beginning training with different datasets"
echo "=========================================="

TRAIN_DATASETS="./data/hospital/train_rag_2000.json ./data/hospital/train_rag_4000.json ./data/hospital/train_rag_6000.json"

EVAL_DATASET="./data/hospital/Qwen_test.json"

CUDA_DEVICES="0,1"

for train_data in $TRAIN_DATASETS; do
    echo ""
    echo "=========================================="
    echo "Begin training with: $train_data"
    echo "=========================================="
    echo ""
    
    if [ ! -f "$train_data" ]; then
        echo "❌ error don't exist: $train_data"
        exit 1
    fi
    
    TORCH_DISTRIBUTED_DEBUG=DETAIL \
    ACCELERATE_DEBUG_VERBOSITY="debug" \
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
    accelerate launch \
        --main_process_port=29919 \
        --mixed_precision=bf16 \
        --dynamo_backend=no \
        --num_machines=1 \
        --num_processes=2 \
        --use_deepspeed \
        finetune_distributed.py \
        --use_lora=True \
        --train-data-path="$train_data" \
        --eval-data-path="$EVAL_DATASET" \
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ training done: $train_data"
        echo ""
    else
        echo ""
        echo "❌ training false: $train_data, exitcode: $?"
        exit 1
    fi

    echo "wait 5 seconds before next training..."
    sleep 5
    echo ""
done

echo ""
echo "=========================================="
echo "all training done"
echo "=========================================="
echo ""
echo "Trained with the following datasets:"
for train_data in $TRAIN_DATASETS; do
    echo "  - $(basename $train_data)"
done
