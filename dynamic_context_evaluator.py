"""
动态上文评估模块
在测试阶段实时维护向量数据库，用历史推理结果作为上文
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
    动态上文评估器
    - 维护历史推理结果的向量数据库
    - 为每张测试图片动态查找相似的历史帧作为上文
    """
    
    def __init__(self, 
                 similarity_threshold=0.8,
                 max_lookback=8,
                 top_k=5,
                 device="cuda"):
        """
        Args:
            similarity_threshold: 相似度阈值，低于此值的参考会被舍弃
            max_lookback: 最多往前查找多少张图片
            top_k: 最多取前k个相似的作为上文
            device: 设备（主模型的设备，CLIP会用CPU）
        """
        self.similarity_threshold = similarity_threshold
        self.max_lookback = max_lookback
        self.top_k = top_k
        self.device = device
        
            # 加载 CLIP 模型用于图片编
        logger.info("加载 CLIP 模型用于动态上文推理（使用GPU）...")
        self.clip_device = "cuda"
        self.clip_model, self.preprocess = clip.load("ViT-L/14", device=self.clip_device)
        self.clip_model.eval()
        
        # 历史记录：存储每张图片的编码和推理结果
        self.history = []  # List of {"image_path": str, "embedding": np.array, "prediction": str}
        
        # 编码缓存：避免重复编码同一张图片
        self.encoding_cache = {}  # {image_path: embedding}
        
        logger.info(f"动态上文评估器初始化完成:")
        logger.info(f"  - 相似度阈值: {similarity_threshold}")
        logger.info(f"  - 最大回溯: {max_lookback}")
        logger.info(f"  - Top-K: {top_k}")
    
    def encode_image(self, image_path: str) -> np.ndarray:
        """
        使用 CLIP 编码图片（带缓存）
        
        Args:
            image_path: 图片路径
        
        Returns:
            归一化的图片向量 (768维)
        """
        # 检查缓存
        if image_path in self.encoding_cache:
            return self.encoding_cache[image_path]
        
        try:
            image = Image.open(image_path).convert('RGB')
            image_input = self.preprocess(image).unsqueeze(0).to(self.clip_device)
            
            with torch.no_grad():
                embedding = self.clip_model.encode_image(image_input)
                # 归一化
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                embedding = embedding.cpu().numpy()[0]  # [768]
            
            # 缓存结果
            self.encoding_cache[image_path] = embedding
            return embedding
        except Exception as e:
            logger.error(f"编码图片失败 {image_path}: {e}")
            return np.zeros(768, dtype=np.float32)
    
    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        计算两个向量的余弦相似度
        
        Args:
            emb1: 向量1
            emb2: 向量2
        
        Returns:
            相似度 (0~1)
        """
        return float(np.dot(emb1, emb2))
    
    def find_similar_context(self, current_image_path: str) -> List[Dict]:
        """
        为当前图片查找相似的历史帧作为上文
        
        Args:
            current_image_path: 当前图片路径
        
        Returns:
            相似历史帧列表 [{"similarity": float, "prediction": str}, ...]
        """
        # 如果没有历史，返回空
        if len(self.history) == 0:
            return []
        
        # 编码当前图片
        current_embedding = self.encode_image(current_image_path)
        
        # 确定查找范围：最多往前看 max_lookback 张
        lookback_count = min(self.max_lookback, len(self.history))
        candidates = self.history[-lookback_count:]  # 取最近的 N 张
        
        # 计算相似度
        similarities = []
        for hist in candidates:
            sim = self.compute_similarity(current_embedding, hist['embedding'])
            if sim >= self.similarity_threshold:
                similarities.append({
                    'similarity': sim,
                    'prediction': hist['prediction'],
                    'image_path': hist['image_path']
                })
        
        # 按相似度降序排序，取 top-k
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        top_references = similarities[:self.top_k]
        
        return top_references
    
    def build_prompt_with_context(self, 
                                   base_prompt_template: str,
                                   references: List[Dict]) -> str:
        """
        根据参考上文构建提示词
        
        Args:
            base_prompt_template: 基础提示词模板（无上文版本）
            references: 参考上文列表
        
        Returns:
            
        """
        if not references:
            # 没有上文，使用简化提示词
            return """你是一个医院安全监控分析员，正在分析最新的监控画面：\n1. 描述：简洁描述你在视频中看到的人员行为和画面内容。\n2. 异常判断：根据描述，判断视频是否存在以下异常情况（仅限当前视频内容）：\n   - 患者摔倒\n   - 患者突然瘫坐不动\n   - 患者倒地不起\n   - 医生正在抢救患者\n   - 患者或家属情绪激动、争执、推搡等冲突行为\n   - 其他常识类异常情况\n\n输出规则：\n- 如果没有异常，必须输出固定格式：`描述：xxx。异常判断：无异常状况。`\n- 如果有异常，必须输出固定格式：`描述：xxx。异常判断：请注意，出现了[具体异常]的情况，需要及时处理或知晓。`\n- **描述** 和 **异常判断** 两部分必须同时出现，缺一不可，且用全角冒号“：”。\n- 输出必须是一段话，不分段、不分条。\n\n例子（有异常）：\n描述：监控中患者从床上摔下来。异常判断：请注意，出现了患者摔倒的情况，需要及时处理或知晓。\n\n例子（无异常）：\n描述：监控中医生正在为患者测量血压。异常判断：无异常状况。"""
        
        # 有上文，构建带历史帧的提示词
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
        将当前图片和预测结果添加到历史记录
        
        Args:
            image_path: 图片路径
            prediction: 模型预测的描述
        """
        embedding = self.encode_image(image_path)
        self.history.append({
            'image_path': image_path,
            'embedding': embedding,
            'prediction': prediction
        })
    
    def reset_history(self):
        """清空历史记录（开始新的测试集时调用）"""
        self.history = []
        logger.info("历史记录已清空")
    
    def get_history_count(self) -> int:
        """获取当前历史记录数量"""
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
    使用动态上文进行评估
    
    Args:
        model: 训练好的模型
        processor: 处理器
        eval_dataset: 测试数据集（ToyDataSet 实例）
        device: 设备
        accelerator: Accelerator 实例
        similarity_threshold: 相似度阈值
        max_lookback: 最大回溯数量
        top_k: Top-K 参考数量
    
    Returns:
        平均损失
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
        logger.info(f"📊 开始动态上文评估 | 总样本数: {len(eval_dataset)}")
        logger.info(f"🔍 相似度阈值: {similarity_threshold} | 最大回溯: {max_lookback} | Top-K: {top_k}")
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
                logger.error(f"评估样本 {idx} 失败: {e}")
                continue
    
    if accelerator.is_local_main_process:
        eval_pbar.close()
    
    avg_loss = eval_loss_sum / max(1, eval_steps)
    
    if accelerator.is_local_main_process:
        logger.info("=" * 80)
        logger.info(f"✅ 动态上文评估完成 | 平均损失: {avg_loss:.4f} | 总样本数: {eval_steps}")
        logger.info("=" * 80)
    
    return avg_loss
