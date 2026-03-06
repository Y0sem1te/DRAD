"""
dynamic context evaluator

This module implements a dynamic context evaluator for the hospital safety m
onitoring task. It maintains a history of past inference results and their corresponding image embeddings, and dynamically retrieves similar historical frames as context for each test image.
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import clip
import numpy as np
from PIL import Image
from typing import List, Dict, Optional
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DynamicContextEvaluator:
    """
    Dynamic Context Evaluator
    - Maintains a vector database of past inference results
    - Dynamically retrieves similar historical frames as context for each test image
    """
    
    def __init__(self, 
                 similarity_threshold=0.8,
                 max_lookback=8,
                 top_k=5,
                 device="cuda"):
        """
        Args:
            similarity_threshold: Similarity threshold, references below this value will be discarded
            max_lookback: Maximum number of images to look back
            top_k: Maximum number of similar images to consider as context
            device: Device (main model's device, CLIP will use CPU)
        """
        self.similarity_threshold = similarity_threshold
        self.max_lookback = max_lookback
        self.top_k = top_k
        self.device = device
        
        logger.info("Loading CLIP model for dynamic context reasoning (using GPU)...")
        self.clip_device = "cuda"
        self.clip_model, self.preprocess = clip.load("ViT-L/14", device=self.clip_device)
        self.clip_model.eval()

        self.history = []  # List of {"image_path": str, "embedding": np.array, "prediction": str}
        self.encoding_cache = {}  # {image_path: embedding}
        
        logger.info(f"Dynamic Context Evaluator initialized:")
        logger.info(f"  - Similarity threshold: {similarity_threshold}")
        logger.info(f"  - Maximum lookback: {max_lookback}")
        logger.info(f"  - Top-K: {top_k}")
    
    def encode_image(self, image_path: str) -> np.ndarray:
        if image_path in self.encoding_cache:
            return self.encoding_cache[image_path]
        
        try:
            image = Image.open(image_path).convert('RGB')
            image_input = self.preprocess(image).unsqueeze(0).to(self.clip_device)
            
            with torch.no_grad():
                embedding = self.clip_model.encode_image(image_input)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                embedding = embedding.cpu().numpy()[0]  # [768]
            
            self.encoding_cache[image_path] = embedding
            return embedding
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {e}")
            return np.zeros(768, dtype=np.float32)
    
    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors
        
        Args:
            emb1: Vector 1
            emb2: Vector 2
        
        Returns:
            Similarity (0~1)
        """
        return float(np.dot(emb1, emb2))
    
    def find_similar_context(self, current_image_path: str) -> List[Dict]:
        """
        Find similar historical frames as context for the current image
        
        Args:
            current_image_path: Path to the current image
        
        Returns:
            Similar historical frames [{"similarity": float, "prediction": str}, ...]
        """
        if len(self.history) == 0:
            return []

        current_embedding = self.encode_image(current_image_path)

        lookback_count = min(self.max_lookback, len(self.history))
        candidates = self.history[-lookback_count:]

        similarities = []
        for hist in candidates:
            sim = self.compute_similarity(current_embedding, hist['embedding'])
            if sim >= self.similarity_threshold:
                similarities.append({
                    'similarity': sim,
                    'prediction': hist['prediction'],
                    'image_path': hist['image_path']
                })

        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        top_references = similarities[:self.top_k]
        
        return top_references
    
    def build_prompt_with_context(self, 
                                   base_prompt_template: str,
                                   references: List[Dict]) -> str:
        """
        Args:
            base_prompt_template: Base prompt template (without context)
            references: List of contextual references
        
        Returns:
            
        """
        if not references:
            return """你是一个医院安全监控分析员，正在分析最新的监控画面：\n1. 描述：简洁描述你在视频中看到的人员行为和画面内容。\n2. 异常判断：根据描述，判断视频是否存在以下异常情况（仅限当前视频内容）：\n   - 患者摔倒\n   - 患者突然瘫坐不动\n   - 患者倒地不起\n   - 医生正在抢救患者\n   - 患者或家属情绪激动、争执、推搡等冲突行为\n   - 其他常识类异常情况\n\n输出规则：\n- 如果没有异常，必须输出固定格式：`描述：xxx。异常判断：无异常状况。`\n- 如果有异常，必须输出固定格式：`描述：xxx。异常判断：请注意，出现了[具体异常]的情况，需要及时处理或知晓。`\n- **描述** 和 **异常判断** 两部分必须同时出现，缺一不可，且用全角冒号“：”。\n- 输出必须是一段话，不分段、不分条。\n\n例子（有异常）：\n描述：监控中患者从床上摔下来。异常判断：请注意，出现了患者摔倒的情况，需要及时处理或知晓。\n\n例子（无异常）：\n描述：监控中医生正在为患者测量血压。异常判断：无异常状况。"""

        history_lines = []
        for i, ref in enumerate(references, 1):
            history_lines.append(f"- 历史帧{i}（相似度{ref['similarity']:.2%}）：{ref['prediction']}")
        
        history_str = "\n".join(history_lines)
        
        prompt = f"""你是一个医院安全监控分析员，正在分析最新的监控画面。
以下内容是当前画面之前的历史监控描述，仅用于提供背景参考，不代表当前画面的事实，不能单独作为异常判断依据：

{history_str}

请重点分析当前画面，以当前画面为主要依据，在理解历史行为发展脉络的基础上，准确描述人员状态、动作或事件变化。
1. 描述：简洁描述你在视频中看到的人员行为和画面内容（仅限当前画面）。
2. 异常判断：根据描述，判断视频是否存在以下异常情况（仅限当前视频内容）：
 - 患者摔倒
 - 患者突然瘫坐不动
 - 患者倒地不起
 - 医生正在抢救患者
 - 患者或家属情绪激动、争执、推搡等冲突行为
 - 其他常识类异常情况

输出规则：
- 如果没有异常，必须输出固定格式：描述：xxx。异常判断：无异常状况。
- 如果有异常，必须输出固定格式：描述：xxx。异常判断：请注意，出现了[具体异常]的情况，需要及时处理或知晓。
- **描述** 和 **异常判断** 两部分必须同时出现，缺一不可，且用全角冒号“：”。
- 输出必须是一段话，不分段、不分条。

例子（有异常）：
描述：监控中患者从床上摔下来。异常判断：请注意，出现了患者摔倒的情况，需要及时处理或知晓。

例子（无异常）：
描述：监控中医生正在为患者测量血压。异常判断：无异常状况。"""
        
        return prompt
    
    def add_to_history(self, image_path: str, prediction: str):
        """
        add inference result to history, including encoding the image and storing the prediction
        
        Args:
            image_path: Path to the image
            prediction: Model prediction for the image
        """
        embedding = self.encode_image(image_path)
        self.history.append({
            'image_path': image_path,
            'embedding': embedding,
            'prediction': prediction
        })
    
    def reset_history(self):
        self.history = []
        logger.info("Dynamic context history has been reset.")
    
    def get_history_count(self) -> int:
        return len(self.history)


def evaluate_with_dynamic_context(model, 
                                   processor,
                                   eval_dataset,
                                   device,
                                   accelerator,
                                   similarity_threshold=0.8,
                                   max_lookback=8,
                                   top_k=5,
                                   gen_max_new_tokens: int = 128):
    """
    
    Args:
        model: model to evaluate
        processor: processor for tokenization and prompt construction
        eval_dataset: evaluation dataset (ToyDataSet instance)
        device: device
        accelerator: Accelerator instance
        similarity_threshold: similarity threshold
        max_lookback: maximum lookback count
        top_k: Top-K reference count
    
    Returns:
        avg loss
    """
    model.eval()

    # NOTE: If you set eval_on_main_process_only=True in multi-GPU runs,
    # non-main processes will largely idle/wait and may time out in some environments.
    # Default is False to keep all processes busy and avoid abnormal termination.
    
    evaluator = DynamicContextEvaluator(
        similarity_threshold=similarity_threshold,
        max_lookback=max_lookback,
        top_k=top_k,
        device=device
    )
    
    eval_loss_sum = 0.0
    eval_steps = 0
    
    if accelerator.is_local_main_process:
        logger.info("=" * 80)
        logger.info(f"📊 Starting dynamic context evaluation | Total samples: {len(eval_dataset)}")
        logger.info(f"🔍 Similarity threshold: {similarity_threshold} | Maximum lookback: {max_lookback} | Top-K: {top_k}")
        logger.info("=" * 80)
    
    if accelerator.is_local_main_process:
        eval_pbar = tqdm(range(len(eval_dataset)), desc="Evaluating", ncols=100)
    else:
        eval_pbar = range(len(eval_dataset))
    
    with torch.no_grad():
        for idx in eval_pbar:
            sample = eval_dataset[idx]
            
            image_path = sample['messages'][0]['content'][0]['image']
            ground_truth = sample['messages'][1]['content'][0]['text']
            references = evaluator.find_similar_context(image_path)
            prompt = evaluator.build_prompt_with_context("", references)

            dynamic_message = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_path},
                            {"type": "text", "text": prompt}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": ground_truth}
                        ]
                    }
                ]
            }
            
            conversation = dynamic_message["messages"]
            text_prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
            
            try:
                image = Image.open(image_path).convert("RGB")
                inputs = processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt")
                inputs = inputs.to(device)
                
                input_ids_list = inputs["input_ids"][0].tolist()
                label_ids = [-100] * len(input_ids_list)
                
                # find（<|im_start|>assistant\n to <|im_end|>\n）
                for i in range(len(input_ids_list) - 2):
                    if input_ids_list[i] == 151644 and input_ids_list[i+1] == 77091 and input_ids_list[i+2] == 198:
                        start_idx = i + 3
                        for j in range(start_idx, len(input_ids_list) - 1):
                            if input_ids_list[j] == 151645 and input_ids_list[j+1] == 198:
                                end_idx = j + 2
                                label_ids[start_idx:end_idx] = input_ids_list[start_idx:end_idx]
                                break
                        break
                
                labels = torch.tensor([label_ids], dtype=torch.int64, device=device)
                
                outputs = model(**inputs, labels=labels)
                loss = outputs.loss
                eval_loss_sum += loss.item()
                eval_steps += 1
                
                messages_gen = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_path},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]

                text_prompt_gen = processor.apply_chat_template(
                    messages_gen, tokenize=False, add_generation_prompt=True
                )
                inputs_gen = processor(text=[text_prompt_gen], images=[image], padding=True, return_tensors="pt").to(device)

                gen_kwargs = {
                    "max_new_tokens": gen_max_new_tokens,
                    "do_sample": True,
                    "num_beams": 1,
                    "repetition_penalty": 1.05,
                    "no_repeat_ngram_size": 0,
                    "temperature": 0.8,
                    "top_p": 0.9,
                    "top_k": 50,
                }

                generated_ids = model.generate(**inputs_gen, **gen_kwargs)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs_gen.input_ids, generated_ids)
                ]
                prediction = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                
                evaluator.add_to_history(image_path, prediction)
                
                if accelerator.is_local_main_process:
                    current_avg_loss = eval_loss_sum / eval_steps
                    eval_pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'avg': f'{current_avg_loss:.4f}',
                        'refs': len(references)
                    })
            
            except Exception as e:
                logger.error(f"📊 Evaluation of sample {idx} failed: {e}")
                continue
    
    if accelerator.is_local_main_process:
        eval_pbar.close()
    
    avg_loss = eval_loss_sum / max(1, eval_steps)
    
    if accelerator.is_local_main_process:
        logger.info("=" * 80)
        logger.info(f"✅ Dynamic context evaluation completed | Average loss: {avg_loss:.4f} | Total samples: {eval_steps}")
        logger.info("=" * 80)
    
    return avg_loss
