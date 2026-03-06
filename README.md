# DRAD

This project is built on top of `Qwen2.5-VL` and provides two major pipelines:

- Image task training/evaluation (hospital scenario, with dynamic-context RAG evaluation)
- Video task training/evaluation (MSR-VTT / MSVD, with frame-level history memory and Teacher Forcing)

---

## 1. Project Structure

```text
DRAD/
├── README.md
├── model/                                # Qwen2.5-VL-3B base model directory (prepare manually)
├── data/
│   ├── hospital/                         # image-task data
│   ├── msr-vtt/                          # video-task data (MSR-VTT)
│   └── MSVD/                             # video-task data (MSVD)
├── preprocess/
│   ├── create_train_data.py
|   └── preprocess_video_frames.py        # offline frame-cache preprocessing
|
├── shell/                                # Shell scripts directory
│   ├── run_preprocess_msrvtt.sh
│   ├── run_video_training_msr_vtt.sh
│   ├── run_video_training_msvd.sh
│   ├── run_image_training_hspt.sh
│   ├── run_train_different_datasets.sh
│   └── run_eval_multiple_lora.sh
├── finetune_distributed.py               # distributed training for image tasks
├── finetune_distributed_video.py         # video training (legacy)
├── finetune_distributed_video_rapid.py   # video training (recommended, with frame cache)
├── eval_metrics.py                       # image evaluation (with RAG)
├── eval_metrics_norag.py                 # image evaluation (without RAG)
├── dynamic_context_evaluator.py          # dynamic context retrieval module
└── util/
    └── logutil.py
```

---

## 2. Environment Setup

Python 3.10/3.11 + CUDA 12.4 + PyTorch 2.5.1 is recommended.

Recommended installation:

```bash
pip install -r requirements.txt
```

If you use the official Qwen VL toolchain, ensure `qwen_vl_utils` is available in your environment.

---

## 3. Model & Data Preparation

### 3.1 Base Model

Place the Qwen2.5-VL-3B base model in `./model` under this project root.
All training/evaluation scripts load from this path by default.

### 3.2 Video Dataset Format (`finetune_distributed_video_rapid.py`)

Each split JSON (`train/eval/test`) must be a list of records. Each record requires:

```json
{
  "video_id": "video_000001",
  "video": "video_000001.mp4",
  "caption": [
    "A man is cooking in the kitchen.",
    "A person prepares food on a stove."
  ]
}
```

Notes:

- `video` must be a path relative to `--video-dir`.
- `caption` can be either a string or a list of strings (both are supported in code).
- `video_id` should be unique within the split.
- Invalid `video` paths are filtered during dataset loading.

Minimal valid record with a single caption string:

```json
{
  "video_id": "video_000002",
  "video": "subsetA/video_000002.mp4",
  "caption": "A woman is walking in a park."
}
```

### 3.3 Image Dataset Format (`finetune_distributed.py`)

Image data uses Qwen-style chat records. Each item should contain `messages` with one `user` turn and one `assistant` turn.

`raw_images` should store the real image files referenced by JSON, for example:

```text
data/hospital/
├── raw_images/
│   ├── main_datasets/
│   │   ├── image1/
│   │   ├── image2/
│   │   └── image3/
│   └── eval_image/
├── Qwen_train.json
├── Qwen_test.json
└── Qwen_eval.json
```

All image paths in JSON must point to files under `raw_images` (or another existing local path).

Example:

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "./data/hospital/raw_images/main_datasets/image1/0001.jpg"},
        {"type": "text", "text": "Describe what is happening in this image."}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "A doctor is checking a patient in a hospital ward."}
      ]
    }
  ]
}
```

Notes:

- In `eval_metrics.py` and `eval_metrics_norag.py`, the same `messages` structure is expected.
- For evaluation files, make sure every sample includes both user image/text and assistant reference text.
- If image files are moved, update JSON paths accordingly; otherwise training/evaluation will fail to read those samples.

### 3.4 Recommended split files and quick validation

Common split naming in this project:

- Video (MSR-VTT):
  - `./data/msr-vtt/msrvtt_train_official.json`
  - `./data/msr-vtt/msrvtt_test_official.json`
  - `./data/msr-vtt/msrvtt_val_official.json`
- Video (MSVD):
  - `./data/MSVD/msvd_train.json`
  - `./data/MSVD/msvd_val.json`
  - `./data/MSVD/msvd_test.json`

Quick checks before training:

- All JSON files are UTF-8 and loadable.
- Every video sample has `video_id`, `video`, and `caption`.
- Every image sample has `messages` with both user and assistant turns.
- Paths in JSON are valid on disk.

---

## 4. Video Frame Cache Preprocessing (Recommended)

Pre-sampling and caching frames into `.npy` files can significantly reduce decoding overhead during training.

```bash
bash run_preprocess_msrvtt.sh
```

Or run manually:

```bash
python preprocess_video_frames.py \
  --json-paths \
    ./data/msr-vtt/msrvtt_train_official.json \
    ./data/msr-vtt/msrvtt_test_official.json \
    ./data/msr-vtt/msrvtt_val_official.json \
  --video-dir ./data/msr-vtt/MSRVTT/videos/all \
  --output-dir ./preprocess/npys \
  --num-frames 12 \
  --overwrite
```

---

## 5. Training

### 5.1 Video Training (Recommended: rapid version)

```bash
CUDA_VISIBLE_DEVICES=0,1 \
TORCH_DISTRIBUTED_DEBUG=DETAIL \
ACCELERATE_DEBUG_VERBOSITY=info \
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
  --eval-json ./data/msr-vtt/msrvtt_test_official.json \
  --test-json ./data/msr-vtt/msrvtt_val_official.json \
  --video-dir ./data/msr-vtt/MSRVTT/videos/all/ \
  --frame-cache-dir ./preprocess/npys \
  --learning-rate 1e-6 \
  --num-frames 12 \
  --max-history 4 \
  --use-teacher-forcing \
  --teacher-forcing-k 3025
```

You can also directly run:

```bash
bash run_video_training_msr_vtt.sh
```

MSVD example:

```bash
bash run_video_training_msvd.sh
```

### 5.2 Image Training

```bash
bash run_image_training_hspt.sh
```

Multi-dataset loop example:

```bash
bash run_train_different_datasets.sh
```

---

## 6. Evaluation

### 6.1 Video Evaluation

`finetune_distributed_video_rapid.py` automatically runs validation at the end of each epoch, then runs final test evaluation after training.

Current metrics include:

- CIDEr
- ROUGE-L
- METEOR
- BLEU-1/2/3/4

For faster evaluation, evaluation is generation-only and does **not** compute eval loss.

### 6.2 Image Evaluation (RAG)

```bash
python eval_metrics.py \
  --lora_dir ./train_output/<time>/epoch_<n> \
  --top_k 5 \
  --max_lookback 8 \
  --similarity_threshold 0.85
```

Batch-evaluate multiple LoRA checkpoints:

```bash
bash run_eval_multiple_lora.sh
```

---

## 7. Output Artifacts

### 7.1 Video Training Outputs

Default output path:

```text
train_output_video/<timestamp>/
```

Common artifacts:

- `epoch_5`, `epoch_10`, ...: periodic checkpoints
- `tensorboard/`: TensorBoard logs
- `lora_loss_history.png` / `full_loss_history.png`: training curves
- `lora_loss_data.json` / `full_loss_data.json`: curve data
- `final_evaluation_results.json`: final test metrics
- `chat_template.json` and processor files

### 7.2 Image Training Outputs

Default output path:

```text
train_output/<timestamp>/
```

The directory has a similar structure (checkpoints and logs) depending on script configuration.

---

## 8. Visualization

TensorBoard logs are written during training:

```bash
tensorboard --logdir ./train_output_video
```

---

## 9. FAQ

1. **`--use_lora` argument behavior**  
   The current script defines it as `type=bool`; pass explicit `True/False` in command lines.

2. **DeepSpeed port conflict**  
   If startup fails, change `--main_process_port` to an available port.

3. **Dataset size unexpectedly reduced**  
   Non-existent video paths are filtered during loading; ensure JSON `video` paths are valid under `--video-dir`.

---

## 10. Acknowledgements

- Qwen2.5-VL / HuggingFace Transformers
- PEFT / DeepSpeed / Accelerate
- COCO Caption Evaluation toolkit
