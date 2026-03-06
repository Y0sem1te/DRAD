"""
Data augmentation script: Add reference text to training set
Features:
1. Read Qwen_train.json and Qwen_test.json
2. For each sample, find up to 5 images with similarity > threshold from the previous 8 samples
3. Add descriptions of similar images as reference text to the prompt
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import json
import torch
import clip
from PIL import Image
from tqdm import tqdm
import copy


device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-L/14", device=device)


def compute_single_embedding(img_path):
    """
    Compute CLIP embedding for a single image
    
    Args:
        img_path: Image file path
    
    Returns:
        torch.Tensor: Normalized embedding vector, None if failed
    """
    try:
        img = Image.open(img_path).convert('RGB')
        img_input = preprocess(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            emb = clip_model.encode_image(img_input)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        
        return emb.cpu()
    
    except Exception as e:
        print(f"Failed to compute embedding ({img_path}): {e}")
        return None


def compute_batch_embeddings(img_paths, batch_size=32):
    """
    Batch compute CLIP embeddings for images (significant speedup)
    
    Args:
        img_paths: List of image file paths
        batch_size: Batch processing size
    
    Returns:
        dict: {img_path: embedding_tensor}
    """
    embeddings = {}
    valid_paths = []
    valid_indices = []
    
    # Filter valid paths
    for i, path in enumerate(img_paths):
        if path and os.path.exists(path):
            valid_paths.append(path)
            valid_indices.append(i)
    
    print(f"Batch computing embeddings for {len(valid_paths)} images...")
    
    for i in tqdm(range(0, len(valid_paths), batch_size), desc="Computing embeddings"):
        batch_paths = valid_paths[i:i + batch_size]
        batch_images = []
        batch_valid_paths = []
        
        # Load images
        for path in batch_paths:
            try:
                img = Image.open(path).convert('RGB')
                batch_images.append(preprocess(img))
                batch_valid_paths.append(path)
            except Exception as e:
                print(f"\nFailed to load image ({path}): {e}")
                continue
        
        if not batch_images:
            continue
        
        # Batch encode
        try:
            batch_tensor = torch.stack(batch_images).to(device)
            with torch.no_grad():
                batch_emb = clip_model.encode_image(batch_tensor)
                batch_emb = batch_emb / batch_emb.norm(dim=-1, keepdim=True)
            
            # Save results
            for path, emb in zip(batch_valid_paths, batch_emb):
                embeddings[path] = emb.cpu()
        
        except Exception as e:
            print(f"\nBatch encoding failed: {e}")
            # Fallback to single image processing
            for path in batch_valid_paths:
                emb = compute_single_embedding(path)
                if emb is not None:
                    embeddings[path] = emb
    
    print(f"Successfully computed {len(embeddings)} embeddings")
    return embeddings


def compute_similarity_from_embeddings(emb1, emb2):
    """
    Compute similarity from pre-computed embeddings (extremely fast)
    
    Args:
        emb1: First embedding tensor
        emb2: Second embedding tensor
    
    Returns:
        float: Similarity score (0~1)
    """
    if emb1 is None or emb2 is None:
        return 0.0
    
    try:
        similarity = (emb1 @ emb2.T).item()
        return similarity
    except Exception as e:
        print(f"Failed to compute similarity: {e}")
        return 0.0


def extract_image_path(data_item):
    """Extract image path from data item"""
    try:
        content = data_item['messages'][0]['content']
        for item in content:
            if item.get('type') == 'image':
                return item.get('image', '')
    except Exception as e:
        print(f"Failed to extract image path: {e}")
    return ''


def extract_assistant_text(data_item):
    """Extract assistant's response text from data item"""
    try:
        content = data_item['messages'][1]['content']
        for item in content:
            if item.get('type') == 'text':
                return item.get('text', '')
    except Exception as e:
        print(f"Failed to extract response text: {e}")
    return ''


def find_similar_references(current_idx, data_list, embeddings_cache, similarity_threshold=0.9):
    """
    Find references with similarity > threshold from the previous 8 samples (using pre-computed embeddings)
    
    Args:
        current_idx: Current sample index
        data_list: Data list
        embeddings_cache: Pre-computed embeddings dictionary
        similarity_threshold: Similarity threshold (default 0.9)
    
    Returns:
        list: List of description texts from similar samples
    """
    if current_idx == 0:
        return []
    
    current_img_path = extract_image_path(data_list[current_idx])
    if not current_img_path or current_img_path not in embeddings_cache:
        return []
    
    current_emb = embeddings_cache[current_img_path]
    
    references = []
    start_idx = max(0, current_idx - 8)
    
    for prev_idx in range(start_idx, current_idx):
        prev_img_path = extract_image_path(data_list[prev_idx])
        
        if not prev_img_path or prev_img_path not in embeddings_cache:
            continue
        
        prev_emb = embeddings_cache[prev_img_path]
        similarity = compute_similarity_from_embeddings(current_emb, prev_emb)
        
        if similarity >= similarity_threshold:
            prev_text = extract_assistant_text(data_list[prev_idx])
            if prev_text:
                references.append({
                    'similarity': similarity,
                    'text': prev_text,
                    'index': prev_idx
                })
    
    return references


def add_reference_to_prompt(data_item, references):
    """
    Add reference text before the original prompt (keep original prompt)
    Args:
        data_item: Data item (will be modified)
        references: List of reference texts
    Returns:
        dict: Modified data item
    """
    if not references:
        return data_item
    history_captions = []
    for i, ref in enumerate(references, 1):
        history_captions.append(f"- 历史帧{i}（相似度{ref['similarity']:.2%}）：{ref['text']}")
    
    history_str = "\n".join(history_captions)
    reference_prefix = f"""你是一个医院安全监控分析员，正在分析最新的监控画面。
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
    
    try:
        content = data_item['messages'][0]['content']
        for item in content:
            if item.get('type') == 'text':
                item['text'] = reference_prefix
                break
    except Exception as e:
        print(f"Failed to add reference text: {e}")
    
    return data_item

def process_dataset(input_path, output_path, similarity_threshold=0.9):
    """
    Process dataset and add reference text
    
    Args:
        input_path: Input JSON file path
        output_path: Output JSON file path
        similarity_threshold: Similarity threshold
    """
    print(f"\nProcessing file: {input_path}")
    print(f"Similarity threshold: {similarity_threshold}")
    print("Reading data...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
    
    print(f"Total {len(data_list)} samples")
    
    # ===== Pre-compute embeddings for all images =====
    print("\n[Step 1/2] Pre-computing embeddings for all images...")
    all_img_paths = [extract_image_path(item) for item in data_list]
    embeddings_cache = compute_batch_embeddings(all_img_paths, batch_size=32)
    print(f"Embeddings computed! Cached {len(embeddings_cache)} vectors\n")
    
    # ===== Process samples =====
    print("[Step 2/2] Finding similar references and generating new data...")
    total_samples = len(data_list)
    samples_with_references = 0
    total_references = 0
    
    new_data_list = []
    
    for idx in tqdm(range(total_samples), desc="Processing samples"):
        data_item = copy.deepcopy(data_list[idx])
        
        references = find_similar_references(idx, data_list, embeddings_cache, similarity_threshold)
        
        if references:
            samples_with_references += 1
            total_references += len(references)
            data_item = add_reference_to_prompt(data_item, references)
        
        new_data_list.append(data_item)
        if (idx + 1) % 1000 == 0:
            print(f"\nProcessed {idx + 1}/{total_samples} samples")
            print(f"  - Samples with references: {samples_with_references}")
            print(f"  - Avg references per sample: {total_references / (idx + 1):.2f}")
    
    print(f"\nSaving results to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_data_list, f, ensure_ascii=False, indent=4)
    
    print(f"\n{'='*60}")
    print(f"Processing completed!")
    print(f"{'='*60}")
    print(f"Total samples: {total_samples}")
    print(f"Samples with references added: {samples_with_references} ({samples_with_references/total_samples*100:.2f}%)")
    print(f"Total references: {total_references}")
    print(f"Avg references per sample: {total_references/total_samples:.2f}")
    print(f"Avg references for samples with refs: {total_references/max(samples_with_references, 1):.2f}")
    print(f"Output file: {output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    BASE_DIR = "."
    SIMILARITY_THRESHOLD = 0.85
    
    print("\n" + "="*60)
    print("Starting to process training set")
    print("="*60)
    process_dataset(
        input_path=f"{BASE_DIR}/Qwen_train.json",
        output_path=f"{BASE_DIR}/Qwen_train_with_ref.json",
        similarity_threshold=SIMILARITY_THRESHOLD
    )

    print("All data processing completed!")
    print(f"New training set: {BASE_DIR}/Qwen_train_with_ref.json")
