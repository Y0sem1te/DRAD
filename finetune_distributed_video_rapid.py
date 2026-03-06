from builtins import Exception, len, range
import torch
import json
import datetime
import os
import matplotlib.pyplot as plt
import random
import copy
import re
import numpy as np
import cv2
from PIL import Image
import decord
from decord import VideoReader, cpu
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# LoRA imports
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from functools import partial
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple

from util.logutil import init_logger, get_logger

from accelerate import Accelerator, DeepSpeedPlugin
import deepspeed

# Evaluation metrics
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.bleu.bleu import Bleu

print("Init deepspeed plugin...")
deepspeed_plugin = DeepSpeedPlugin(
    zero_stage=2,
    gradient_accumulation_steps=2,
    zero3_save_16bit_model=True,
    offload_optimizer_device="none",
    offload_param_device="none"
)
print("Init deepspeed plugin done")

print("Init accelerator...")
accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
print("Init accelerator done")

device = accelerator.device
output_dir = f'train_output_video/{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}/'

if accelerator.is_local_main_process:
    os.makedirs(output_dir, exist_ok=True)
    init_logger(output_dir)
    logger = get_logger()


class VideoDataset(Dataset):
    
    def __init__(self, json_path: str, video_dir: str, num_frames: int = 12, frame_cache_dir: Optional[str] = None):
        """
        Args:
            json_path: JSON file path
            video_dir: Video directory
            num_frames: Number of frames sampled per video
        """
        super().__init__()
        with open(json_path, "r") as f:
            self.data = json.load(f)
        self.video_dir = video_dir
        self.num_frames = num_frames
        self.frame_cache_dir = frame_cache_dir
        
        valid_data = []
        for item in self.data:
            video_path = os.path.join(self.video_dir, item['video'])
            if os.path.exists(video_path):
                valid_data.append(item)
        
        self.data = valid_data
        if accelerator.is_local_main_process:
            logger.info(f"Loaded {len(self.data)} valid video samples from {json_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        video_path = os.path.join(self.video_dir, item['video'])
        frames = self.sample_frames(video_path, video_rel_path=item['video'])
        return {
            'video_id': item['video_id'],
            'video': item['video'],
            'caption': item['caption'],
            'frames': frames,
        }

    def get_cache_path(self, video_rel_path: str) -> str:
        normalized_rel = video_rel_path.lstrip("/")
        return os.path.join(self.frame_cache_dir, f"{normalized_rel}.nf{self.num_frames}.npy")

    def load_cached_frames(self, video_rel_path: str) -> Optional[List[Image.Image]]:
        if not self.frame_cache_dir:
            return None
        cache_path = self.get_cache_path(video_rel_path)
        if not os.path.exists(cache_path):
            return None
        try:
            raw = np.load(cache_path)
            if raw.ndim != 4 or raw.shape[0] == 0:
                return None
            if raw.shape[0] < self.num_frames:
                pad = np.repeat(raw[-1:], self.num_frames - raw.shape[0], axis=0)
                raw = np.concatenate([raw, pad], axis=0)
            elif raw.shape[0] > self.num_frames:
                raw = raw[:self.num_frames]
            return [Image.fromarray(raw[i]) for i in range(self.num_frames)]
        except Exception as e:
            print(f"[WARNING] Failed to load frame cache, fallback to decode: {cache_path} | error: {e}")
            return None
    
    def sample_frames(self, video_path: str, video_rel_path: Optional[str] = None) -> List[Image.Image]:
        """
        extract num_frames frames from the video at video_path
        
        Args:
            video_path: video file path
        
        Returns:
            Sampled frames list (PIL Images)
        """
        if video_rel_path is not None:
            cached = self.load_cached_frames(video_rel_path)
            if cached is not None:
                return cached

        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
        except Exception as e:
            print(f"[WARNING] Failed to open video, using blank frames: {video_path} | error: {e}")
            return [Image.new('RGB', (224, 224), color=(0, 0, 0)) for _ in range(self.num_frames)]

        if total_frames == 0:
            print(f"[WARNING] Video has 0 frames, using blank frames: {video_path}")
            return [Image.new('RGB', (224, 224), color=(0, 0, 0)) for _ in range(self.num_frames)]

        if total_frames <= self.num_frames:
            indices = list(range(total_frames)) + [total_frames - 1] * (self.num_frames - total_frames)
        else:
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int).tolist()

        raw = vr.get_batch(indices).asnumpy()  # (T, H, W, 3) uint8 RGB
        frames = [Image.fromarray(raw[i]) for i in range(len(indices))]
        return frames

class MemoryBank:
    def __init__(self, max_history: int = 3):
        self.history = []
        self.max_history = max_history
    
    def add(self, predicted_text: str):
        """Add a new history record"""
        self.history.append(predicted_text)
    
    def retrieve(self) -> List[str]:
        """
        Retrieve the most recent historical frame texts (up to k)
        Returns:
            List of recent historical frame predicted texts
        """
        return self.history[-self.max_history:]
    
    def clear(self):
        self.history = []
    
    def __len__(self):
        return len(self.history)



def compute_teacher_forcing_epsilon(iteration: int, k: float) -> float:
    """
    Compute the probability epsilon for Teacher Forcing (inverse Sigmoid decay)
    Formula: epsilon = k / (k + exp(iteration/k))
    Initially epsilon ≈ 1 (almost always use GT), later epsilon ≈ 0 (almost always use prediction)
    """
    import math
    epsilon = k / (k + math.exp(iteration / k))
    return epsilon


def find_assistant_content_sublist_indexes(l):
    """Find the positions of assistant content in input_ids"""
    start_indexes = []
    end_indexes = []
    
    for i in range(len(l) - 2):
        if l[i] == 151644 and l[i+1] == 77091 and l[i+2] == 198:
            start_indexes.append(i+3)
            for j in range(i+3, len(l)-1):
                if l[j] == 151645 and l[j+1] == 198:
                    end_indexes.append(j+2)
                    break
    
    return list(zip(start_indexes, end_indexes))


def build_prompt_with_context(history_events: List[str]) -> str:
    if not history_events:
        return "Analyze the current video frame and describe the main event happening in the video."
    
    timeline_lines = []
    for i, event in enumerate(history_events):
        timeline_lines.append(f"- Time step {i+1}: {event}")
    
    history_str = "\n".join(timeline_lines)
    
    prompt = f"""The following is a chronological sequence of events that have occurred in the video so far:
{history_str}

Task: Based on the full history of events listed above and the current visual content, describe the **entire event** captured in the video from the beginning up to now.
Ensure your description summarizes the main action performed in the video."""
    
    return prompt


def train_one_video_sample(
    model,
    processor,
    frames: List[Image.Image],
    captions: List[str],
    device: str,
    num_frames: int = 12,
    accelerator = None,
    teacher_forcing_epsilon: float = 1.0,
    max_history: int = 3
) -> Tuple[float, List[str], int]:
    """
    Train a single video sample (iteratively generate by time steps, compute weighted loss)
    Args:
        model: Qwen model
        processor: Processor
        frames: Sampled T frames images
        captions: All reference descriptions of the video
        device: Device
        num_frames: Number of frames T
        accelerator: Accelerator object for immediate backpropagation
        teacher_forcing_epsilon: Teacher Forcing probability (probability of using GT)
    Returns:
        total_loss_value: Weighted total loss value (float, without gradient)
        num_generated: Actual number of generate calls
    """
    memory = MemoryBank(max_history=max_history)
    raw_weights = [((t + 1) / num_frames) for t in range(num_frames)]
    sum_weights = sum(raw_weights)
    norm_weights = [w / sum_weights for w in raw_weights]
    total_loss_value = 0.0
    num_generated = 0
    accumulated_loss = 0.0
    backward_interval = 2
    ground_truth_for_loss = random.choice(captions)
    
    for t in range(num_frames):
        current_frame = frames[t]
        context_texts = memory.retrieve()
        prompt = build_prompt_with_context(context_texts)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": current_frame},
                    {"type": "text", "text": prompt}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": ground_truth_for_loss}
                ]
            }
        ]
        
        text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        inputs = processor(text=[text_prompt], images=[current_frame], padding=True, return_tensors="pt")
        inputs = inputs.to(device, non_blocking=True)
        
        input_ids_list = inputs["input_ids"][0].tolist()
        label_ids = [-100] * len(input_ids_list)
        
        for begin, end in find_assistant_content_sublist_indexes(input_ids_list):
            label_ids[begin:end] = input_ids_list[begin:end]
        labels = torch.tensor([label_ids], dtype=torch.int64, device=device)
        outputs = model(**inputs, labels=labels)
        loss_t = outputs.loss * norm_weights[t]

        accumulated_loss += loss_t
        total_loss_value += loss_t.item()
        
        should_backward = ((t + 1) % backward_interval == 0) or (t == num_frames - 1)
        if should_backward:
            if accelerator is not None:
                accelerator.backward(accumulated_loss)
            else:
                accumulated_loss.backward()
            accumulated_loss = 0.0
        
        # Teacher Forcing
        use_ground_truth = random.random() < teacher_forcing_epsilon
        
        if use_ground_truth:
            prediction_t = random.choice(captions)
            memory.add(prediction_t)
            del outputs, loss_t, inputs, labels
        else:
            num_generated += 1
            with torch.no_grad():
                messages_gen_train = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": current_frame},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
                text_prompt_gen_train = processor.apply_chat_template(messages_gen_train, tokenize=False, add_generation_prompt=True)
                inputs_gen_train = processor(
                    text=[text_prompt_gen_train], images=[current_frame], padding=True, return_tensors="pt"
                )
                inputs_gen_train = inputs_gen_train.to(device, non_blocking=True)
                gen_kwargs = {
                    "max_new_tokens": 48,
                    "do_sample": False,
                    "temperature": None
                }
                generated_ids = model.generate(**inputs_gen_train, **gen_kwargs)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs_gen_train.input_ids, generated_ids)
                ]
                prediction_t = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
            
            memory.add(prediction_t)
            del outputs, loss_t, inputs, labels, inputs_gen_train, generated_ids
        
        # torch.cuda.empty_cache()
    
    return total_loss_value, num_generated


def write_chat_template(processor, output_dir):
    output_chat_template_file = os.path.join(output_dir, "chat_template.json")
    chat_template_json_string = json.dumps({"chat_template": processor.chat_template}, indent=2, sort_keys=True) + "\n"
    with open(output_chat_template_file, "w", encoding="utf-8") as writer:
        writer.write(chat_template_json_string)
        if accelerator.is_local_main_process:
            logger.info(f"chat template saved in {output_chat_template_file}")


def setup_lora_config():
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none",
        use_rslora=False,
        use_dora=False,
    )
    return lora_config


def apply_lora_to_model(model):
    lora_config = setup_lora_config()
    model = get_peft_model(model, lora_config)
    return model


def plot_training_loss(loss_history, epoch_losses, output_dir, training_type="LoRA", eval_metrics=None):
    num_plots = 2
    if eval_metrics is not None:
        num_plots += 1
    
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 5 * num_plots))
    if num_plots == 1:
        axes = [axes]
    
    ax_idx = 0
    
    # Plot 1: Step-wise loss
    axes[ax_idx].plot(loss_history, label=f'{training_type} Step Loss', alpha=0.7)
    axes[ax_idx].set_xlabel('Step')
    axes[ax_idx].set_ylabel('Loss')
    axes[ax_idx].set_title(f'{training_type} Training Loss Over Steps')
    axes[ax_idx].grid(True, alpha=0.3)
    axes[ax_idx].legend()
    ax_idx += 1
    
    # Plot 2: Epoch-wise train loss
    if epoch_losses:
        axes[ax_idx].plot(range(1, len(epoch_losses) + 1), epoch_losses, 
                         label=f'{training_type} Epoch Train Loss', marker='o', linewidth=2)
        axes[ax_idx].set_xlabel('Epoch')
        axes[ax_idx].set_ylabel('Average Train Loss')
        axes[ax_idx].set_title(f'{training_type} Average Training Loss per Epoch')
        axes[ax_idx].grid(True, alpha=0.3)
        axes[ax_idx].legend()
        ax_idx += 1
    
    # Plot 3: Eval metrics
    if eval_metrics is not None:
        for metric_name, values in eval_metrics.items():
            if len(values) > 0:
                axes[ax_idx].plot(range(1, len(values) + 1), values,
                                label=metric_name, marker='o', linewidth=2)
        axes[ax_idx].set_xlabel('Epoch')
        axes[ax_idx].set_ylabel('Score')
        axes[ax_idx].set_title('Evaluation Metrics per Epoch')
        axes[ax_idx].grid(True, alpha=0.3)
        axes[ax_idx].legend()
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{training_type.lower()}_loss_history.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    loss_data = {
        'training_type': training_type,
        'step_losses': loss_history,
        'epoch_losses': epoch_losses,
        'eval_metrics': eval_metrics,
    }
    
    json_path = os.path.join(output_dir, f'{training_type.lower()}_loss_data.json')
    with open(json_path, 'w') as f:
        json.dump(loss_data, f, indent=2)


def evaluate_video(
    model,
    processor,
    eval_dataset: VideoDataset,
    device: str,
    accelerator,
    num_frames: int = 12,
    max_history: int = 3
) -> Dict[str, float]:
    """
    Evaluate video dataset (generate-only; compute CIDEr/ROUGE-L/METEOR/BLEU)
    
    Returns:
        metrics: Dictionary of evaluation metrics {'CIDEr': x, 'ROUGE_L': x, 'METEOR': x, 'BLEU_4': x}
    """
    model.eval()
    
    gts = {}  # {video_id: [ref1, ref2, ...]}
    res = {}  # {video_id: [prediction]}
    
    if accelerator.is_local_main_process:
        logger.info("=" * 80)
        logger.info(f"Starting video evaluation | Total samples: {len(eval_dataset)}")
        logger.info("=" * 80)
        eval_pbar = tqdm(range(len(eval_dataset)), desc="Evaluating", ncols=100)
    else:
        eval_pbar = range(len(eval_dataset))
    
    with torch.no_grad():
        for idx in eval_pbar:
            sample = eval_dataset[idx]
            video_id = sample['video_id']
            video_path = os.path.join(eval_dataset.video_dir, sample['video'])
            captions = sample['caption']
            # Ensure captions is a list
            if isinstance(captions, str):
                captions = [captions]
            frames = eval_dataset.sample_frames(video_path, video_rel_path=sample['video'])
            
            memory = MemoryBank(max_history=max_history)
            final_prediction = ""

            for t in range(num_frames):
                current_frame = frames[t]
                context_texts = memory.retrieve()
                
                prompt = build_prompt_with_context(context_texts)

                messages_gen = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": current_frame},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]

                text_prompt_gen = processor.apply_chat_template(messages_gen, tokenize=False, add_generation_prompt=True)

                inputs_gen = processor(
                    text=[text_prompt_gen], images=[current_frame], padding=True, return_tensors="pt"
                ).to(device, non_blocking=True)

                gen_kwargs = {
                    "max_new_tokens": 48,
                    "do_sample": False,
                    "temperature": None
                }
                generated_ids = model.generate(**inputs_gen, **gen_kwargs)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs_gen.input_ids, generated_ids)
                ]
                prediction_t = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                
                memory.add(prediction_t)
                
                del generated_ids, generated_ids_trimmed, inputs_gen

                if t == num_frames - 1:
                    final_prediction = prediction_t
            
            if isinstance(captions, str):
                gts[video_id] = [captions]
            else:
                gts[video_id] = captions if isinstance(captions, list) else [captions]
            res[video_id] = [final_prediction]
            
            if accelerator.is_local_main_process:
                eval_pbar.set_postfix({'sample': idx + 1})
    
    if accelerator.is_local_main_process:
        eval_pbar.close()
    
    metrics = {}
    if accelerator.is_local_main_process and len(gts) > 0:
        try:
            # Tokenize predictions and ground truths
            gts_formatted = {}
            for vid, caps in gts.items():
                cap_list = caps if isinstance(caps, list) else [caps]
                gts_formatted[vid] = [{'caption': str(c)} for c in cap_list]

            res_formatted = {}
            for vid, caps in res.items():
                cap_list = caps if isinstance(caps, list) else [caps]
                res_formatted[vid] = [{'caption': str(c)} for c in cap_list]

            tokenizer = PTBTokenizer()
            gts_tokenized = tokenizer.tokenize(gts_formatted)
            res_tokenized = tokenizer.tokenize(res_formatted)
            
            # CIDEr
            cider_scorer = Cider()
            cider_score, _ = cider_scorer.compute_score(gts_tokenized, res_tokenized)
            metrics['CIDEr'] = cider_score
            
            # ROUGE-L
            rouge_scorer = Rouge()
            rouge_score, _ = rouge_scorer.compute_score(gts_tokenized, res_tokenized)
            metrics['ROUGE_L'] = rouge_score
            
            # METEOR
            meteor_scorer = Meteor()
            meteor_score, _ = meteor_scorer.compute_score(gts_tokenized, res_tokenized)
            metrics['METEOR'] = meteor_score
            
            # BLEU (1-4)
            bleu_scorer = Bleu(4)
            bleu_scores, _ = bleu_scorer.compute_score(gts_tokenized, res_tokenized)
            metrics['BLEU_1'] = bleu_scores[0]
            metrics['BLEU_2'] = bleu_scores[1]
            metrics['BLEU_3'] = bleu_scores[2]
            metrics['BLEU_4'] = bleu_scores[3]
            
            logger.info("=" * 80)
            logger.info("calculate the metrics:")
            logger.info(f"  CIDEr: {metrics['CIDEr']:.4f}")
            logger.info(f"  ROUGE-L: {metrics['ROUGE_L']:.4f}")
            logger.info(f"  METEOR: {metrics['METEOR']:.4f}")
            logger.info(f"  BLEU-1: {metrics['BLEU_1']:.4f}")
            logger.info(f"  BLEU-2: {metrics['BLEU_2']:.4f}")
            logger.info(f"  BLEU-3: {metrics['BLEU_3']:.4f}")
            logger.info(f"  BLEU-4: {metrics['BLEU_4']:.4f}")
            logger.info("=" * 80)
        except Exception as e:
            logger.error(f"fail to calculate metrics: {e}")
            metrics = {'CIDEr': 0.0, 'ROUGE_L': 0.0, 'METEOR': 0.0, 'BLEU_4': 0.0}
    
    model.train()
    return metrics

def train(
    train_json_path: str,
    eval_json_path: str,
    test_json_path: str,
    video_dir: str,
    num_frames: int = 12,
    use_lora: bool = True,
    seed: int = 42,
    use_teacher_forcing: bool = True,
    teacher_forcing_k: float = 700,
    learning_rate: float = 1e-6,
    frame_cache_dir: Optional[str] = "./main_ex_pic"
):
    """
    train function for distributed video finetuning on video dataset
    
    Args:
        train_json_path: train set JSON path
        eval_json_path: eval set JSON path (evaluated at the end of each epoch)
        test_json_path: test set JSON path (final evaluation after training)
        video_dir: directory of video files
        num_frames: number of frames sampled per video
        use_lora: whether to use LoRA
        max_history: maximum number of historical frames
        seed: random seed
        use_teacher_forcing: whether to use Teacher Forcing strategy
        teacher_forcing_k: Teacher Forcing decay rate parameter (larger means slower decay)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if accelerator.is_local_main_process:
        logger.info("Loading model...")
    model_load_kwargs = {
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "flash_attention_2",
    }
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "./model", **model_load_kwargs
        )
        if accelerator.is_local_main_process:
            logger.info("Using attn_implementation=flash_attention_2")
    except Exception as e:
        if accelerator.is_local_main_process:
            logger.warning(f"flash_attention_2 unavailable, fallback to default attention: {e}")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "./model", torch_dtype=torch.bfloat16
        )
    
    if use_lora:
        if accelerator.is_local_main_process:
            logger.info("Applying LoRA to the model...")
        model = apply_lora_to_model(model)
    else:
        if accelerator.is_local_main_process:
            logger.info("Using full fine-tuning (LoRA disabled)")
    
    processor = AutoProcessor.from_pretrained(
        "./model", min_pixels=128*28*28, max_pixels=224*28*28, padding_side="right"
    )
    
    train_dataset = VideoDataset(train_json_path, video_dir, num_frames=num_frames, frame_cache_dir=frame_cache_dir)
    eval_dataset = VideoDataset(eval_json_path, video_dir, num_frames=num_frames, frame_cache_dir=frame_cache_dir)
    test_dataset = VideoDataset(test_json_path, video_dir, num_frames=num_frames, frame_cache_dir=frame_cache_dir)

    if accelerator.is_local_main_process and frame_cache_dir:
        logger.info(f"Frame cache dir enabled: {frame_cache_dir}")

    def collate_video_batch(batch):
        return {
            'video_id': [item['video_id'] for item in batch],
            'video': [item['video'] for item in batch],
            'caption': [item['caption'] for item in batch],
            'frames': [item['frames'] for item in batch],
        }

    dataloader_workers = min(8, os.cpu_count() or 4)
    use_persistent_workers = dataloader_workers > 0
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        pin_memory=True,
        num_workers=dataloader_workers,
        persistent_workers=use_persistent_workers,
        prefetch_factor=4,
        collate_fn=collate_video_batch,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=dataloader_workers,
        persistent_workers=use_persistent_workers,
        prefetch_factor=4,
        collate_fn=collate_video_batch,
    )
    
    epochs = 20
    lr = learning_rate if use_lora else 1e-5
    
    if accelerator.is_local_main_process:
        logger.info(f"Learning rate: {lr}")
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    model, optimizer, train_loader, eval_loader = accelerator.prepare(
        model, optimizer, train_loader, eval_loader
    )
    
    loss_history = []
    epoch_losses = []
    eval_metrics_history = {
        'CIDEr': [], 'ROUGE_L': [], 'METEOR': [], 'BLEU_4': []
    }
    
    writer = None
    if accelerator.is_local_main_process:
        tensorboard_dir = os.path.join(output_dir, 'tensorboard')
        writer = SummaryWriter(log_dir=tensorboard_dir)
        logger.info(f"TensorBoard logs will be saved to: {tensorboard_dir}")
        logger.info(f"To view, run: tensorboard --logdir={tensorboard_dir}")
    
    global_step = 0
    
    try:
        for epoch in range(epochs):
            model.train()
            epoch_loss_sum = 0.0
            epoch_steps = 0
            
            if accelerator.is_local_main_process:
                logger.info("=" * 80)
                logger.info(f"Epoch {epoch + 1}/{epochs}")
                logger.info("=" * 80)
                train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Training", ncols=100)
            else:
                train_pbar = train_loader
            
            for batch in train_pbar:
                video_id = batch['video_id'][0]
                captions = batch['caption'][0]
                # Ensure captions is a list
                if isinstance(captions, str):
                    captions = [captions]
                frames = batch['frames'][0]
                
                if use_teacher_forcing:
                    current_epsilon = compute_teacher_forcing_epsilon(global_step, k=teacher_forcing_k)
                else:
                    current_epsilon = 0.0
                
                with accelerator.accumulate(model):
                    total_loss_value, num_generated = train_one_video_sample(
                        model, processor, frames, captions,
                        device, num_frames,
                        accelerator=accelerator,
                        teacher_forcing_epsilon=current_epsilon,
                        max_history=args.max_history
                    )
                    
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    if accelerator.is_local_main_process:
                        current_loss = total_loss_value
                        loss_history.append(current_loss)
                        epoch_loss_sum += current_loss
                        epoch_steps += 1
                        
                        if writer is not None:
                            writer.add_scalar('Train/step_loss', current_loss, global_step)
                            writer.add_scalar('Train/teacher_forcing_epsilon', current_epsilon, global_step)
                            writer.add_scalar('Train/num_generated', num_generated, global_step)
                        global_step += 1
                        
                        avg_loss_so_far = epoch_loss_sum / epoch_steps
                        train_pbar.set_postfix({
                            'loss': f'{current_loss:.4f}',
                            'avg': f'{avg_loss_so_far:.4f}',
                            'ε': f'{current_epsilon:.3f}'
                        })
            
            if accelerator.is_local_main_process:
                train_pbar.close()
            
            accelerator.wait_for_everyone()
            
            if accelerator.is_local_main_process:
                logger.info("="*80)
                logger.info("Starting end-of-epoch evaluation")
                logger.info("="*80)
            
            metrics = evaluate_video(
                model, processor, eval_dataset,
                device, accelerator, num_frames, max_history=args.max_history
            )

            if accelerator.is_local_main_process:
                for key in ['CIDEr', 'ROUGE_L', 'METEOR', 'BLEU_4']:
                    if key in metrics:
                        eval_metrics_history[key].append(metrics[key])
            
            accelerator.wait_for_everyone()
            
            if accelerator.is_local_main_process and epoch_steps > 0:
                avg_epoch_loss = epoch_loss_sum / epoch_steps
                epoch_losses.append(avg_epoch_loss)
                
                if writer is not None:
                    writer.add_scalar('Train/epoch_loss', avg_epoch_loss, epoch + 1)
                    
                    for metric_name, metric_value in metrics.items():
                        writer.add_scalar(f'Eval/{metric_name}', metric_value, epoch + 1)
                
                logger.info("=" * 80)
                logger.info(f"Epoch {epoch+1}/{epochs} completed")
                logger.info(f"   Training average Loss: {avg_epoch_loss:.6f}")
                logger.info("=" * 80)
            
            if (epoch + 1) % 5 == 0:
                epoch_ckpt_dir = f"{output_dir}epoch_{epoch+1}"
                
                if accelerator.is_local_main_process:
                    logger.info(f"Saving checkpoint for epoch {epoch+1} to {epoch_ckpt_dir}")
                
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
            
                try:
                    import deepspeed
                    lora_params = [p for n, p in unwrapped_model.named_parameters() if "lora" in n]

                    if len(lora_params) == 0:
                        if accelerator.is_local_main_process:
                            logger.warning("No LoRA params found. Falling back to full save.")
                        unwrapped_model.save_pretrained(
                            epoch_ckpt_dir,
                            is_main_process=accelerator.is_main_process,
                            save_function=accelerator.save,
                        )
                    else:
                        with deepspeed.zero.GatheredParameters(lora_params, modifier_rank=0):
                            if accelerator.is_local_main_process:
                                unwrapped_model.save_pretrained(
                                    epoch_ckpt_dir,
                                    is_main_process=accelerator.is_main_process,
                                    save_function=accelerator.save,
                                )
                except Exception as e:
                    if accelerator.is_local_main_process:
                        logger.exception("DeepSpeed gather failed, fallback to normal save. Error: %s", e)
                    unwrapped_model.save_pretrained(
                        epoch_ckpt_dir,
                        is_main_process=accelerator.is_main_process,
                        save_function=accelerator.save,
                    )
            

    finally:
        if accelerator.is_local_main_process and len(loss_history) > 0:
            logger.info("Plotting loss curves...")
            training_type = "LoRA" if use_lora else "Full"
            plot_training_loss(
                loss_history, epoch_losses, output_dir, training_type,
                eval_metrics=eval_metrics_history
            )
        
        if writer is not None:
            writer.close()
            if accelerator.is_local_main_process:
                logger.info("TensorBoard writer closed.")
        
        if writer is not None:
            writer.close()
            if accelerator.is_local_main_process:
                logger.info("TensorBoard writer closed.")
    
    accelerator.wait_for_everyone()
    
    if accelerator.is_local_main_process:
        logger.info("=" * 80)
        logger.info("complete training. Starting final test evaluation...")
        logger.info("=" * 80)
    
    final_test_metrics = evaluate_video(
        model, processor, test_dataset,
        device, accelerator, num_frames, max_history=args.max_history
    )
    
    if accelerator.is_local_main_process:
        final_results = {
            'final_test_metrics': final_test_metrics,
            'training_config': {
                'num_frames': num_frames,
                'use_lora': use_lora,
                'learning_rate': lr,
                'epochs_trained': len(epoch_losses),
            }
        }
        
        results_path = os.path.join(output_dir, 'final_evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        if writer is not None:
            for metric_name, metric_value in final_test_metrics.items():
                writer.add_scalar(f'Test/{metric_name}', metric_value, 0)
        
        logger.info("=" * 80)
        logger.info("Final test evaluation results:")
        logger.info(f"  CIDEr: {final_test_metrics.get('CIDEr', 0.0):.4f}")
        logger.info(f"  ROUGE-L: {final_test_metrics.get('ROUGE_L', 0.0):.4f}")
        logger.info(f"  METEOR: {final_test_metrics.get('METEOR', 0.0):.4f}")
        logger.info(f"  BLEU-1: {final_test_metrics.get('BLEU_1', 0.0):.4f}")
        logger.info(f"  BLEU-2: {final_test_metrics.get('BLEU_2', 0.0):.4f}")
        logger.info(f"  BLEU-3: {final_test_metrics.get('BLEU_3', 0.0):.4f}")
        logger.info(f"  BLEU-4: {final_test_metrics.get('BLEU_4', 0.0):.4f}")
        logger.info(f"  Results saved to: {results_path}")
        logger.info("=" * 80)
        
        logger.info("Training completed.")
        processor.save_pretrained(output_dir)
        write_chat_template(processor, output_dir)
    
    return output_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train Qwen2.5-VL on video dataset')
    parser.add_argument('--use_lora', type=bool, default=True, help='Use LoRA or full fine-tuning')
    parser.add_argument('--train-json', type=str, required=True, help='Training dataset JSON path')
    parser.add_argument('--eval-json', type=str, required=True, help='Validation dataset JSON path (for epoch evaluation)')
    parser.add_argument('--test-json', type=str, required=True, help='Test dataset JSON path (for final evaluation)')
    parser.add_argument('--video-dir', type=str, required=True, help='Video directory')
    parser.add_argument('--frame-cache-dir', type=str, required=True, help='Directory for preprocessed frame cache (.npy).')
    parser.add_argument('--num-frames', type=int, default=12, help='Number of frames to sample per video')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--max-history', type=int, default=4, help='Maximum number of historical frames to use as context')
    parser.add_argument('--use-teacher-forcing', action='store_true', help='Enable teacher forcing strategy')
    parser.add_argument('--teacher-forcing-k', type=float, default=200, help='Teacher forcing decay rate (larger = slower decay)')
    parser.add_argument('--learning-rate', type=float, default=1e-6, help='Learning rate for training')
    
    args = parser.parse_args()
    
    train(
        train_json_path=args.train_json,
        eval_json_path=args.eval_json,
        test_json_path=args.test_json,
        video_dir=args.video_dir,
        num_frames=args.num_frames,
        use_lora=args.use_lora,
        seed=args.seed,
        use_teacher_forcing=args.use_teacher_forcing,
        teacher_forcing_k=args.teacher_forcing_k,
        learning_rate=args.learning_rate,
        frame_cache_dir=args.frame_cache_dir
    )