### multi gpu ###
### ***NOTE***: if set --main_process_port=0 (according to accelerate documentation, it will automatically choose a port number. but it seems not well implemented in deepspeed, if set 0, process hangs). so we need to specify a port number manually.
### with rag
TORCH_DISTRIBUTED_DEBUG=DETAIL \
ACCELERATE_DEBUG_VERBOSITY="debug" \
CUDA_VISIBLE_DEVICES="0,1" \
accelerate launch \
  --main_process_port=29919 \
  --mixed_precision=bf16 \
  --dynamo_backend=no \
  --num_machines=1 \
  --num_processes=2 \
  --use_deepspeed \
  finetune_distributed.py \
  --train-data-path='./data/hospital/Qwen_train_with_ref.json' \
  --eval-data-path='./data/hospital/Qwen_test.json' \
  --use_lora=True \
  --use_rag=True