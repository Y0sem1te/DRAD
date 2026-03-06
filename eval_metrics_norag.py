import nltk
import json
import os
import torch
import jieba
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider import Cider
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from pprint import pprint
from peft import PeftModel
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider import Cider
from datetime import datetime
chencherry = SmoothingFunction()
scorer_rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
cider_scorer = Cider()

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
    extract from messages:
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
    extract standard answer from text of assistant
    """
    for msg in messages:
        if msg["role"] == "assistant":
            for c in msg["content"]:
                if c["type"] == "text":
                    return c["text"]
    return None

def run_model_batch(batch_items: list) -> list:
    """
    Batch inference to improve GPU utilization
    batch_items: list of (image_path, user_prompt)
    Returns: list of predictions
    """
    batch_messages = []
    for image_path, user_prompt in batch_items:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": user_prompt},
                ],
            }
        ]
        batch_messages.append(messages)
    
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in batch_messages
    ]
    all_image_inputs = []
    all_video_inputs = []
    for msg in batch_messages:
        image_inputs, video_inputs = process_vision_info(msg)
        all_image_inputs.extend(image_inputs if image_inputs else [])
        all_video_inputs.extend(video_inputs if video_inputs else [])
    inputs = processor(
        text=texts,
        images=all_image_inputs if all_image_inputs else None,
        videos=all_video_inputs if all_video_inputs else None,
        padding=True,
        return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        gen_kwargs = {
            "max_new_tokens": 128,
            "do_sample": True,
            "temperature": 0.8,
            "top_p": 0.9,
            "repetition_penalty": 1.2,
        }
        generated_ids = model.generate(**inputs, **gen_kwargs)

    predictions = []
    for i, gen_id in enumerate(generated_ids):
        trimmed = gen_id[len(inputs.input_ids[i]):]
        out_text = processor.decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        predictions.append(out_text.strip())
    
    return predictions

def process_dataset(json_path, hyp_out="hyp.txt", ref_out="ref.txt", batch_size=4, skip_ref=False):
    with open(json_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    hyp_f = open(hyp_out, "w", encoding="utf-8")
    if not skip_ref:
        ref_f = open(ref_out, "w", encoding="utf-8")
        for item in dataset:
            messages = item["messages"]
            ref_text = extract_assistant_text(messages)
            assert ref_text is not None, "Standard answer from assistant not found"
            ref_f.write(ref_text.strip() + "\n")
        ref_f.close()
        print(f"Reference file generated: {ref_out}")

    batch_items = []
    for item in dataset:
        messages = item["messages"]
        image_path, user_text = extract_user_info(messages)
        assert image_path is not None, "User image not found"
        assert user_text is not None, "User text prompt not found"
        batch_items.append((image_path, user_text))

    all_predictions = []
    total_batches = (len(batch_items) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(batch_items), batch_size), desc="Batch inference", total=total_batches):
        batch = batch_items[i:i + batch_size]
        predictions = run_model_batch(batch)
        all_predictions.extend(predictions)
    
    for pred in all_predictions:
        pred_clean = pred.replace('\n', ' ').replace('\r', ' ')
        hyp_f.write(pred_clean + "\n")
    
    hyp_f.close()
    print(f"Prediction file generated: {hyp_out}")

def save_log_multi(log_file, lora_dir, all_results, avg_result):
    """
    all_results: list of dict (10 evaluation results)
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

    base_model = "./model"
    
    # method1: Evaluate a single epoch.
    lora_dir = "./train_output/20260121160303/epoch_11"
    lora_dirs = [lora_dir]
    
    # method2: Evaluate all epochs in a training output directory.
    # output_root = "./train_output/20260121160303"
    # lora_dirs = sorted(
    #     [os.path.join(output_root, d) for d in os.listdir(output_root)
    #      if d.startswith("epoch_")],
    #     key=lambda x: int(x.split("_")[-1])
    # )

    log_file = os.path.join(os.path.dirname(lora_dirs[0]), "evaluation_log.txt")

    for lora_dir in lora_dirs:
        print(f"\n==========================")
        print(f"Evaluating model: {lora_dir}")
        print(f"==========================\n")

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
        ref_file = f"./dataset/ref_{os.path.basename(lora_dir)}.txt"
        BATCH_SIZE = 6 

        for run_idx in range(EVAL_REPEAT):
            print(f"Run {run_idx + 1}/{EVAL_REPEAT}...")
            print(f"Using batch inference, batch_size={BATCH_SIZE}\n")

            hyp_file = f"./dataset/hyp_{os.path.basename(lora_dir)}_run{run_idx + 1}.txt"

            process_dataset(
                "./Qwen_eval.json",
                hyp_file,
                ref_file,
                batch_size=BATCH_SIZE,
                skip_ref=(run_idx > 0)
            )

            hyp_list = load_txt_file(hyp_file)
            ref_list = load_txt_file(ref_file)

            assert len(hyp_list) == len(ref_list), "The two files must have the same number of lines!"

            single_result = evaluate(hyp_list, ref_list)

            all_results.append(single_result)

        metrics = all_results[0].keys()
        avg_result = {m: np.mean([r[m] for r in all_results]) for m in metrics}

        print("\n====== Average Evaluation Results for Current Model (10 runs) ======")
        print(f"Model: {lora_dir}")
        for k, v in avg_result.items():
            print(f"{k}: {v:.4f}")

        # ---- Save logs (10 runs + average) ----
        save_log_multi(log_file, lora_dir, all_results, avg_result)

        print(f"\n Log saved to: {log_file}\n")

