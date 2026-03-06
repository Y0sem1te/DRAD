import nltk
import json
import os
import torch
import jieba
import numpy as np
import clip
import argparse
from PIL import Image
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider import Cider
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from pprint import pprint
from peft import PeftModel
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

chencherry = SmoothingFunction()
scorer_rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
cider_scorer = Cider()


class SimpleVectorDB:
    def __init__(self):
        self.embeddings = []  # List[np.ndarray]
        self.predictions = []  # List[str]
        self.image_paths = []  # List[str]
    
    def add(self, embedding: np.ndarray, image_path: str, prediction: str):
        """add a record"""
        self.embeddings.append(embedding.flatten())
        self.predictions.append(prediction)
        self.image_paths.append(image_path)
    
    def search_recent(self, query_embedding: np.ndarray, max_lookback: int = 8, 
                     top_k: int = 5, similarity_threshold: float = 0.8):
        """search similar records from recent entries"""
        if not self.embeddings:
            return []
        
        # Get the most recent max_lookback records
        start_idx = max(0, len(self.embeddings) - max_lookback)
        recent_embeddings = np.array(self.embeddings[start_idx:])
        
        if len(recent_embeddings) == 0:
            return []
        
        # Calculate cosine similarity
        query = query_embedding.flatten()
        similarities = np.dot(recent_embeddings, query)
        
        # Filter and sort
        results = []
        for i, sim in enumerate(similarities):
            if sim >= similarity_threshold:
                actual_idx = start_idx + i
                results.append({
                    'similarity': float(sim),
                    'prediction': self.predictions[actual_idx],
                    'image_path': self.image_paths[actual_idx]
                })
        
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]
    
    def reset(self):
        self.embeddings = []
        self.predictions = []
        self.image_paths = []

def tokenize_chinese(text):
    return list(jieba.cut(text))

def load_txt_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]
    
def rouge_l_score(ref, hyp):
    """Calculate Chinese ROUGE-L using Longest Common Subsequence (LCS)"""
    ref_tokens = list(jieba.cut(ref))
    hyp_tokens = list(jieba.cut(hyp))
    m = len(ref_tokens)
    n = len(hyp_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n] 
    # ROUGE-L F1
    recall = lcs / m
    precision = lcs / n
    if recall + precision == 0:
        return 0.0
    f1 = 2 * recall * precision / (recall + precision)
    return f1

def evaluate(hyp_list, ref_list):
    bleu1_scores = []
    bleu4_scores = []
    rouge_scores = []
    meteor_scores = []

    hyp_dict = {}
    ref_dict = {}

    for i, (hyp, ref) in enumerate(zip(hyp_list, ref_list)):

        hyp_tokens = tokenize_chinese(hyp)
        ref_tokens = [tokenize_chinese(ref)]

        # BLEU-1
        bleu1_scores.append(
            sentence_bleu(ref_tokens, hyp_tokens,
                          weights=(1, 0, 0, 0),
                          smoothing_function=chencherry.method1)
        )
        # BLEU-4
        bleu4_scores.append(
            sentence_bleu(ref_tokens, hyp_tokens,
                          weights=(0.25, 0.25, 0.25, 0.25),
                          smoothing_function=chencherry.method1)
        )
        # ROUGE-L
        rouge = rouge_l_score(ref, hyp)
        rouge_scores.append(rouge)
        # METEOR
        meteor = meteor_score(ref_tokens, hyp_tokens)
        meteor_scores.append(meteor)
        # CIDEr 
        hyp_dict[i] = [" ".join(hyp_tokens)]
        ref_dict[i] = [" ".join(tokenize_chinese(ref))]

    cider_avg, _ = cider_scorer.compute_score(ref_dict, hyp_dict)

    return {
        "BLEU-1": sum(bleu1_scores) / len(bleu1_scores),
        "BLEU-4": sum(bleu4_scores) / len(bleu4_scores),
        "ROUGE-L": sum(rouge_scores) / len(rouge_scores),
        "CIDEr": cider_avg,
        "METEOR": sum(meteor_scores) / len(meteor_scores),
    }

def extract_user_info(messages):
    """
    Extract from messages (e.g., messages1 / messages2):
    - image_path
    - user_prompt (text)
    """
    image_path = None
    user_prompt = None

    # messages [ { "role":..., "content": [...] } ]
    for msg in messages:
        if msg["role"] == "user":
            for c in msg["content"]:
                if c["type"] == "image":
                    image_path = c["image"]
                elif c["type"] == "text":
                    user_prompt = c["text"]

    return image_path, user_prompt

def extract_assistant_text(messages):
    """
    Extract the standard answer text from the assistant
    """
    for msg in messages:
        if msg["role"] == "assistant":
            for c in msg["content"]:
                if c["type"] == "text":
                    return c["text"]
    return None


def build_prompt_with_rag(references):
    """Build prompt with historical context based on RAG retrieval results"""
    if not references:
        return """你是一个医院安全监控分析员，正在分析最新的监控画面：\n1. 描述：简洁描述你在视频中看到的人员行为和画面内容。\n2. 异常判断：根据描述，判断视频是否存在以下异常情况（仅限当前视频内容）：\n   - 患者摔倒\n   - 患者突然瘫坐不动\n   - 患者倒地不起\n   - 医生正在抢救患者\n   - 患者或家属情绪激动、争执、推搡等冲突行为\n   - 其他常识类异常情况\n\n输出规则：\n- 如果没有异常，必须输出固定格式：`描述：xxx。异常判断：无异常状况。`\n- 如果有异常，必须输出固定格式：`描述：xxx。异常判断：请注意，出现了[具体异常]的情况，需要及时处理或知晓。`\n- **描述** 和 **异常判断** 两部分必须同时出现，缺一不可，且用全角冒号"："。\n- 输出必须是一段话，不分段、不分条。\n\n例子（有异常）：\n描述：监控中患者从床上摔下来。异常判断：请注意，出现了患者摔倒的情况，需要及时处理或知晓。\n\n例子（无异常）：\n描述：监控中医生正在为患者测量血压。异常判断：无异常状况。"""

    history_lines = []
    for i, ref in enumerate(references, 1):
        history_lines.append(f"- 历史帧{i}（相似度{ref['similarity']:.2%}）：{ref['prediction']}")
    history_str = "\n".join(history_lines)

    return f"""你是一个医院安全监控分析员，正在分析最新的监控画面。
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
- **描述** 和 **异常判断** 两部分必须同时出现，缺一不可，且用全角冒号"："。
- 输出必须是一段话，不分段、不分条。

例子（有异常）：
描述：监控中患者从床上摔下来。异常判断：请注意，出现了患者摔倒的情况，需要及时处理或知晓。

例子（无异常）：
描述：监控中医生正在为患者测量血压。异常判断：无异常状况。"""


def batch_encode_images_clip(image_paths, clip_model, clip_preprocess, device, batch_size=32):
    """Batch encode all images"""
    print(f"Batch encoding {len(image_paths)} images...")
    embeddings_dict = {}
    
    def load_image(path):
        try:
            img = Image.open(path).convert('RGB')
            return str(path), clip_preprocess(img)
        except Exception as e:
            print(f"Failed to load {path}: {e}")
            return str(path), None
    
    preprocessed = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(load_image, path): path for path in image_paths}
        for future in tqdm(as_completed(futures), total=len(image_paths), desc="Loading"):
            path_str, tensor = future.result()
            if tensor is not None:
                preprocessed[path_str] = tensor
    
    # Batch encode
    print("  → Batch encoding features...")
    paths_list = list(preprocessed.keys())
    tensors_list = [preprocessed[p] for p in paths_list]
    
    with torch.no_grad():
        for i in tqdm(range(0, len(tensors_list), batch_size), desc="Encoding"):
            batch_tensors = tensors_list[i:i+batch_size]
            batch_paths = paths_list[i:i+batch_size]
            
            batch_input = torch.stack(batch_tensors).to(device)
            embeddings = clip_model.encode_image(batch_input)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            embeddings = embeddings.cpu().numpy()
            
            for path, emb in zip(batch_paths, embeddings):
                embeddings_dict[path] = emb
    
    print(f"Done, encoded {len(embeddings_dict)} images")
    return embeddings_dict


def run_model_with_rag(image_path: str, vector_db: SimpleVectorDB, 
                       embeddings_cache: dict, max_lookback: int = 8,
                       top_k: int = 5, similarity_threshold: float = 0.8) -> str:
    """Run model inference with RAG"""
    
    # 1. Get the CLIP encoding of the current image from the cache
    current_embedding = embeddings_cache.get(str(image_path))
    if current_embedding is None:
        print(f"Warning: Encoding not found for {image_path}")
        references = []
    else:
        # 2. Search similar historical frames from the vector database
        references = vector_db.search_recent(
            current_embedding,
            max_lookback=max_lookback,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )
    
    # 3. Build prompt with RAG
    prompt_text = build_prompt_with_rag(references)
    
    # 4. Prepare model input
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        gen_kwargs = {
            "max_new_tokens": 128,
            "do_sample": True,
            "num_beams": 4,
            "temperature": 0.8,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1,
        }
        generated_ids = model.generate(**inputs, **gen_kwargs)
    trimmed = generated_ids[0][len(inputs.input_ids[0]):]
    out_text = processor.decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    prediction = out_text.strip()
    
    # 5. Add current prediction to the vector database
    if current_embedding is not None:
        vector_db.add(current_embedding, str(image_path), prediction)
    
    return prediction

def process_dataset_with_rag(json_path, clip_model, clip_preprocess, clip_device,
                            hyp_out="hyp.txt", ref_out="ref.txt",
                            max_lookback=8, top_k=5, similarity_threshold=0.8):
    with open(json_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # 1. collect all image paths
    print("\nCollecting all image paths...")
    all_image_paths = []
    for item in dataset:
        image_path = item['messages'][0]['content'][0]['image']
        if os.path.exists(image_path):
            all_image_paths.append(image_path)

    print(f"Found {len(all_image_paths)} valid images")
    
    # 2. Batch encode all images
    print("\nPrecomputing CLIP features for all images...")
    embeddings_cache = batch_encode_images_clip(
        all_image_paths, clip_model, clip_preprocess, clip_device, batch_size=32
    )
    
    # 3. Inference with RAG for each image
    vector_db = SimpleVectorDB()
    print("\nStarting inference with RAG...")

    os.makedirs(os.path.dirname(hyp_out), exist_ok=True)
    os.makedirs(os.path.dirname(ref_out), exist_ok=True)
    hyp_f = open(hyp_out, "w", encoding="utf-8")
    ref_f = open(ref_out, "w", encoding="utf-8")

    for item in tqdm(dataset, desc="Processing samples"):
        messages = item["messages"]
        image_path = messages[0]['content'][0]['image']
        ref_text = extract_assistant_text(messages)
        assert ref_text is not None, "No assistant reference text found"

        ref_f.write(ref_text.strip() + "\n")
        try:
            hyp_text = run_model_with_rag(
                image_path, vector_db, embeddings_cache,
                max_lookback=max_lookback,
                top_k=top_k,
                similarity_threshold=similarity_threshold
            ).strip()
            hyp_text = hyp_text.replace('\n', ' ').replace('\r', ' ')
            if not hyp_text:
                hyp_text = "Generation failed"
                print(f"Warning: {image_path} generated empty text")
        except Exception as e:
            hyp_text = f"Generation error: {str(e)[:50]}"
            print(f"Error: {image_path} generation failed: {e}")
        
        hyp_f.write(hyp_text + "\n")

    hyp_f.close()
    ref_f.close()
    print(f"\nGenerated prediction file: {hyp_out}")
    print(f"Generated reference file: {ref_out}")

def save_log_multi(log_file, lora_dir, all_results, avg_result):
    """
    all_results: list of dict (multiple evaluation results)
    avg_result: dict (average results)
    """
    with open(log_file, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 50 + "\n")
        f.write(f"Model: {lora_dir}\n")
        f.write(f"Number of evaluations: {len(all_results)}\n\n")

        for i, r in enumerate(all_results):
            f.write(f"[Result {i+1}]\n")
            for k, v in r.items():
                f.write(f"  {k}: {v:.4f}\n")
            f.write("\n")

        f.write("------ Average Results ------\n")
        for k, v in avg_result.items():
            f.write(f"{k}: {v:.4f}\n")

        f.write("=" * 50 + "\n\n")

EVAL_REPEAT = 3

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='evaluate model with RAG')
    parser.add_argument('--lora_dir', default="./train_output/20260201230911/epoch_14",type=str, help='LoRA weight directory path')
    parser.add_argument('--top_k', type=int, default=5, help='RAG retrieves top-k most similar historical frames (default: 5)')
    parser.add_argument('--max_lookback', type=int, default=8, help='Maximum number of frames to look back (default: 8)')
    parser.add_argument('--similarity_threshold', type=float, default=0.85, help='Similarity threshold (default: 0.85)')
    args = parser.parse_args()

    base_model = "./model"
    lora_dir = args.lora_dir
    lora_dirs = [lora_dir]
    
    # method2: evaluate all epochs
    # output_root = "./train_output/20260123000734/test"
    # lora_dirs = sorted(
    #     [os.path.join(output_root, d) for d in os.listdir(output_root)
    #      if d.startswith("epoch_")],
    #     key=lambda x: int(x.split("_")[-1])
    # )
    if not os.path.exists(lora_dir):
        raise ValueError(f"LoRA directory does not exist: {lora_dir}")

    log_file = os.path.join(os.path.dirname(lora_dirs[0]), f"evaluation_log_rag_k{args.top_k}.txt")

    # Load CLIP model (for RAG)
    clip_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP model for RAG...")
    clip_model, clip_preprocess = clip.load("ViT-L/14", device=clip_device)
    clip_model.eval()

    for lora_dir in lora_dirs:
        print(f"\n{'='*60}")
        print(f"Evaluating model (with RAG): {lora_dir}")
        print(f"{'='*60}\n")

        print("Loading base model...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model,
            torch_dtype="bfloat16",
            device_map="auto"
        )

        print("Loading LoRA:", lora_dir)
        model = PeftModel.from_pretrained(model, lora_dir)
        model = model.merge_and_unload()
        model.eval()

        processor = AutoProcessor.from_pretrained(
            base_model,
            min_pixels=128*28*28,
            max_pixels=256*28*28,
            padding_side="left"
        )

        all_results = []

        for run_idx in range(EVAL_REPEAT):
            print(f"\nRun {run_idx+1}/{EVAL_REPEAT}...")
            print(f"Parameters: top_k={args.top_k}, max_lookback={args.max_lookback}, similarity_threshold={args.similarity_threshold}")

            hyp_file = f"./dataset/hyp_rag_{os.path.basename(lora_dir)}_run{run_idx+1}.txt"
            ref_file = f"./dataset/ref_rag_{os.path.basename(lora_dir)}_run{run_idx+1}.txt"

            process_dataset_with_rag(
                "./Qwen_eval.json",
                clip_model, clip_preprocess, clip_device,
                hyp_file,
                ref_file,
                max_lookback=args.max_lookback,
                top_k=args.top_k,
                similarity_threshold=args.similarity_threshold
            )

            hyp_list = load_txt_file(hyp_file)
            ref_list = load_txt_file(ref_file)

            assert len(hyp_list) == len(ref_list), f"Number of lines in prediction file and reference file do not match! Prediction file ({hyp_file}): {len(hyp_list)} lines, Reference file ({ref_file}): {len(ref_list)} lines. Please check if there is an issue in the data generation process."

            single_result = evaluate(hyp_list, ref_list)

            all_results.append(single_result)
            
            print(f"Current run result: BLEU-1={single_result['BLEU-1']:.4f}, BLEU-4={single_result['BLEU-4']:.4f}")

        # ---- Calculate average results ----
        metrics = all_results[0].keys()
        avg_result = {m: np.mean([r[m] for r in all_results]) for m in metrics}

        print(f"\n{'='*60}")
        print(f"Average evaluation results for current model ({EVAL_REPEAT} runs, with RAG)")
        print(f"{'='*60}")
        print(f"Model: {lora_dir}")
        for k, v in avg_result.items():
            print(f"{k}: {v:.4f}")

        # ---- Save logs (multiple runs + average) ----
        save_log_multi(log_file, lora_dir, all_results, avg_result)

        print(f"\nLogs have been saved to: {log_file}\n")