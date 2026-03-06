from builtins import Exception, len, range
import torch
import json
import datetime
import os
import matplotlib.pyplot as plt
import random
import copy
import re

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# LoRA imports
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from functools import partial
from tqdm import tqdm

from util.logutil import init_logger, get_logger

from dynamic_context_evaluator import evaluate_with_dynamic_context

from accelerate import Accelerator, DeepSpeedPlugin
import deepspeed
# from eval_metrics import QwenEvaluator

print("Init deepspeed plugin...")
# Create a DeepSpeedPlugin configuration object to customize DeepSpeed integration settings。
deepspeed_plugin = DeepSpeedPlugin(
    zero_stage=2,   # Enable ZeRO (Zero Redundancy Optimizer) stage 3 optimization
                    # ZeRO stages: 
                    # 0 - disabled
                    # 1 - optimizer state partitioning
                    # 2 - optimizer state + gradient partitioning
                    # 3 - optimizer state + gradient + parameter partitioning (most memory efficient)
    gradient_accumulation_steps=2,  # Accumulate gradients over 2 steps before optimization
    zero3_save_16bit_model=True,    # Save models in 16-bit precision when using ZeRO stage 3
                                    # Reduces model checkpoint size by 50% while maintaining model quality
    offload_optimizer_device="cpu", # Offload optimizer computation to CPU to drastically reduce GPU memory usage
    offload_param_device="cpu"      # Offload model parameters to CPU to further decrease GPU memory consumption
)
print("Init deepspeed plugin done")
# Initialize the Hugging Face Accelerator with DeepSpeed integration
# Accelerator provides a unified interface for distributed training across various backends
# (TPU, multi-GPU, DeepSpeed, etc.) while maintaining compatibility with PyTorch code
print("Init accelerator...")
accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
print("Init accelerator done")

'''
With the above configuration, when launching the script with below command:
$TORCH_DISTRIBUTED_DEBUG=DETAIL ACCELERATE_DEBUG_VERBOSITY="debug" CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --main_process_port=29919 --mixed_precision=bf16 --dynamo_backend=no --num_machines=1 --num_processes=4 --use_deepspeed finetune_distributed.py 

The final DeepSpeed configuration required will be generated during the subsequent execution of accelerator.prepare(). The configuration details are as follows:

json = {
    "train_batch_size": 8, 
    "train_micro_batch_size_per_gpu": 1, 
    "gradient_accumulation_steps": 2, 
    "zero_optimization": {
        "stage": 3, 
        "offload_optimizer": {
            "device": "cpu", 
            "nvme_path": null
        }, 
        "offload_param": {
            "device": "cpu", 
            "nvme_path": null
        }, 
        "stage3_gather_16bit_weights_on_model_save": true
    }, 
    "gradient_clipping": 1.0, 
    "steps_per_print": inf, 
    "bf16": {
        "enabled": true
    }, 
    "fp16": {
        "enabled": false
    }, 
    "zero_allow_untested_optimizer": true
}
'''

'''
Attention: 
In DeepSpeed, fp16 and bf16 are generally indicative of mixed precision training. 
The half-precision is used for forward and backward computations, while fp32 is used for optimizer computation.
'''

device = accelerator.device
output_dir = f'train_output/{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}/'

if accelerator.is_local_main_process:
    os.makedirs(output_dir, exist_ok=True)
    init_logger(output_dir)
    logger = get_logger()


class ToyDataSet(Dataset): # for toy demo, for train_data/data.json
    def __init__(self, data_path):
        super().__init__()
        with open(data_path, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.0):
        """
        Args:
            patience (int): if loss dont't improve for 'patience' epochs, then stop training
            min_delta (float): minimum change in the monitored quantity to qualify as an improvement. This is used to avoid stopping for very small fluctuations in loss.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0

    def step(self, val_loss):
        """
        return three action：
            - "none": normal step, loss has not improved but patience not reached yet
            - "save": loss improved → save model
            - "stop": loss has not improved and patience reached → stop training
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return "save"
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return "stop"
            return "none"
    
def find_assistant_content_sublist_indexes(l):
    '''
    A message from train_data/data.json may look like below:
        {
            "messages": [
                {'role': 'user', 'content': [{'type': 'image', 'image': 'train_data/1.jpeg'}, {'type': 'text', 'text': '描述一下这个图片'}]}, 
                {'role': 'assistant', 'content': [{'type': 'text', 'text': '这张图片展示了一位年轻女子和她的狗在海滩上玩耍的场景。女子穿着格子衬衫和黑色裤子，坐在沙滩上，与她的金毛犬互动。她们的手臂伸展着，似乎在进行某种游戏或训练。背景是广阔的海洋和晴朗的天空，阳光洒在沙滩上，营造出温暖而宁静的氛围。整体画面充满了快乐和放松的感觉。'}]}
            ]
        }
    After apply_chat_template, the text will look like below:
        ['<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>描述一下这个图片<|im_end|>\n<|im_start|>assistant\n这张图片展示了一位年轻女子和她的狗在海滩上玩耍的场景。女子穿着格子衬衫和黑色裤子，坐在沙滩上，与她的金毛犬互动。她们的手臂伸展着，似乎在进行某种游戏或训练。背景是广阔的海洋和晴朗的天空，阳光洒在沙滩上，营造出温暖而宁静的氛围。整体画面充满了快乐和放松的感觉。<|im_end|>\n']

    This function tries to find the indexes of the assistant content in the input_ids list to build labels.
    '''

    start_indexes = []
    end_indexes = []

    # Iterate through the list to find starting points
    for i in range(len(l) - 2):
        # Check if the current and next elements form the start sequence
        if l[i] == 151644 and l[i+1] == 77091 and l[i+2] == 198:
            start_indexes.append(i+3)
            # Now look for the first 151645 and 198 after the start
            for j in range(i+3, len(l)-1):
                if l[j] == 151645 and l[j+1] == 198:
                    end_indexes.append(j+2) # **NOTE** the <|im_end|>\n 2 tokens should be included in the label, so that model can predicate end of output.
                    break  # Move to the next start after finding the end

    return list(zip(start_indexes, end_indexes))

def collate_fn(batch, processor, device):
    messages = [m['messages'] for m in batch]
    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in messages]
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    input_ids_lists = inputs['input_ids'].tolist()
    assert len(messages) == len(input_ids_lists)

    labels_list = []
    for ids_list in input_ids_lists:
        label_ids = [-100] * len(ids_list) # -100 is the ignore index in loss function
        for begin_end_indexs in find_assistant_content_sublist_indexes(ids_list):
            label_ids[begin_end_indexs[0]:begin_end_indexs[1]] = ids_list[begin_end_indexs[0]:begin_end_indexs[1]]
        labels_list.append(label_ids)

    labels_ids = torch.tensor(labels_list, dtype=torch.int64)
    return inputs, labels_ids

def write_chat_template(processor, output_dir):
    '''
    ***Note**

    We should have not had this function, as normal processor.save_pretrained(output_dir) would save chat_template.json file.
    However, on 2024/09/05, I think a commit introduced a bug to "huggingface/transformers", which caused the chat_template.json file not to be saved. 
    See the below commit, src/transformers/processing_utils.py line 393, this commit avoided chat_template.json to be saved.
    https://github.com/huggingface/transformers/commit/43df47d8e78238021a4273746fc469336f948314#diff-6505546ec5a9ab74b2ce6511681dd31194eb91e9fa3ce26282e487a5e61f9356

    To walk around that bug, we need manually save the chat_template.json file.

    I hope this bug will be fixed soon and I can remove this function then.
    '''
    
    output_chat_template_file = os.path.join(output_dir, "chat_template.json")
    chat_template_json_string = json.dumps({"chat_template": processor.chat_template}, indent=2, sort_keys=True) + "\n"
    with open(output_chat_template_file, "w", encoding="utf-8") as writer:
        writer.write(chat_template_json_string)
        logger.info(f"chat template saved in {output_chat_template_file}")

def setup_lora_config():
    """
    Setup LoRA configuration for the model.
    
    Returns:
        LoraConfig: LoRA configuration object
    """
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        # r=8,  # LoRA rank 
        # lora_alpha=16,  # LoRA alpha parameter
        r=16,
        lora_alpha=32, 
        lora_dropout=0.15,  # Dropout probability for LoRA layers
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ],
        bias="none",
        use_rslora=False, 
        use_dora=False,
    )
    
    return lora_config


def apply_lora_to_model(model):
    """
    Apply LoRA to the model.
    
    Args:
        model: The base model

    Returns:
        model: Model with LoRA applied
    """
    lora_config = setup_lora_config()
    model = get_peft_model(model, lora_config)
    return model

def plot_training_loss(
    loss_history, 
    epoch_losses, 
    output_dir, 
    training_type="LoRA",
    eval_losses=None
):
    """
    Plot training loss history (step loss), training epoch loss, and optional eval loss.
    """

    if eval_losses is not None:
        fig, axes = plt.subplots(3, 1, figsize=(12, 14))
        ax1, ax2, ax3 = axes
    else:
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        ax1, ax2 = axes

    # -----------------------------
    #  1: step-wise loss
    # -----------------------------
    ax1.plot(loss_history, label=f'{training_type} Step Loss', alpha=0.7)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{training_type} Training Loss Over Steps')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # -----------------------------
    #  2: epoch-wise train loss
    # -----------------------------
    if epoch_losses:
        ax2.plot(
            range(1, len(epoch_losses) + 1),
            epoch_losses,
            label=f'{training_type} Epoch Train Loss',
            marker='o',
            linewidth=2
        )
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Average Train Loss')
        ax2.set_title(f'{training_type} Average Training Loss per Epoch')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

    # -----------------------------
    #  3: eval loss curve
    # -----------------------------
    if eval_losses is not None:
        ax3.plot(
            range(1, len(eval_losses) + 1),
            eval_losses,
            label=f'{training_type} Eval Loss',
            marker='s',
            linewidth=2
        )
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Eval Loss')
        ax3.set_title(f'{training_type} Evaluation Loss per Epoch')
        ax3.grid(True, alpha=0.3)
        ax3.legend()

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(output_dir, f'{training_type.lower()}_loss_history.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # -----------------------------
    # save eval_losses to JSON 
    # -----------------------------
    loss_data = {
        'training_type': training_type,
        'step_losses': loss_history,
        'epoch_losses': epoch_losses,
        'eval_losses': eval_losses,     
    }

    json_path = os.path.join(output_dir, f'{training_type.lower()}_loss_data.json')
    with open(json_path, 'w') as f:
        json.dump(loss_data, f, indent=2)

def evaluate(model, eval_loader):
    """Run evaluation and return avg loss."""
    model.eval()
    eval_loss_sum = 0.0
    eval_steps = 0

    with torch.no_grad():
        for batch in eval_loader:
            inputs, labels = batch
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            eval_loss_sum += loss.item()
            eval_steps += 1

    return eval_loss_sum / max(1, eval_steps)

def step(self, val_loss):
    """
    use the validation loss to determine whether to save the model or stop training.
    """
    if val_loss < self.best_loss - self.min_delta:
        self.best_loss = val_loss
        self.counter = 0  # reset
    else:
        self.counter += 1
    return self.counter >= self.patience

def train(
    train_data_path: str,
    eval_data_path: str,
    use_lora: bool = True,
    use_rag: bool = True,
    seed: int = 42,
):
    """
    Train the model with LoRA or full fine-tuning.
    
    Args:
        train_data_path (str): Path to the training data JSON file.
        eval_data_path (str): Path to the evaluation data JSON file.
        use_lora (bool): If True, use LoRA fine-tuning. If False, use full fine-tuning.
        use_rag (bool): If True, use RAG evaluation logic. If False, use simple evaluation.
    """
    # Load the model on the available device(s)
    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     "Qwen/Qwen2.5-VL-3B-Instruct",
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2",
    #     device_map="auto",
    # )
    print("Loading model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "./model", torch_dtype="bfloat16"
    )

    # Apply LoRA to the model based on parameter
    if use_lora:
        if accelerator.is_local_main_process:
            logger.info("Applying LoRA to the model...")
        model = apply_lora_to_model(model)
    else:
        if accelerator.is_local_main_process:
            logger.info("Using full fine-tuning (LoRA disabled)")
        model = model  # No LoRA applied
    
    processor = AutoProcessor.from_pretrained("./model", min_pixels=128*28*28, max_pixels=256*28*28, padding_side="right")
    
    train_dataset = ToyDataSet(train_data_path)
    eval_dataset = ToyDataSet(eval_data_path)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        collate_fn=partial(collate_fn, processor=processor, device=device)
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=1,
        collate_fn=partial(collate_fn, processor=processor, device=device)
    )

    # model.train()
    epochs = 100
    
    # Use different learning rate for LoRA vs full fine-tuning
    if use_lora:
        assert hasattr(model, 'peft_config'), "Model should have peft_config when using LoRA"
        # Higher learning rate for LoRA training
        # lr = 2e-4
        lr = 1e-6
        # lr = 2e-6
        if accelerator.is_local_main_process:
            logger.info(f"Using LoRA training with learning rate: {lr}")
    else:
        # Standard learning rate for full fine-tuning
        lr = 1e-5
        if accelerator.is_local_main_process:
            logger.info(f"Using full fine-tuning with learning rate: {lr}")
    
    # optimizer = AdamW(model.parameters(), lr=lr)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model, optimizer, train_loader, eval_loader = accelerator.prepare(model, optimizer, train_loader, eval_loader)
    
    # Initialize loss tracking
    loss_history = []
    epoch_losses = []
 
    eval_losses = []  
    early_stopper = EarlyStopping(patience=3, min_delta=5e-4)

    # all_metrics = {"BLEU-1": [], "BLEU-4": [],"ROUGE-L": [],"CIDEr": []}

    try:
        for epoch in range(epochs):
            steps = 0
            epoch_loss_sum = 0.0
            epoch_steps = 0
            total_batches = len(train_loader)
            total_global_samples = total_batches * accelerator.num_processes

            if accelerator.is_local_main_process:
                logger.info("=" * 80)
                logger.info(f".  Begin epoch {epoch + 1}/{epochs}")
                logger.info(f"   Batch for each gpu: {total_batches}")
                logger.info(f"   Total global samples: {total_global_samples}")
                logger.info("=" * 80)

            model.train()
            
            train_pbar = (
                tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Training", total=total_batches, ncols=100)
                if accelerator.is_local_main_process
                else train_loader
            )

            for batch in train_pbar:
                steps += 1
                with accelerator.accumulate(model):
                    inputs, labels = batch
                    outputs = model(**inputs, labels=labels)
                    loss = outputs.loss
                    accelerator.backward(loss)
                    # If use deepseed,`accelerator.backward(loss)` is doing that automatically. Therefore, this function will not work. 
                    # For detail, see https://github.com/huggingface/accelerate/blob/main/src/accelerate/utils/deepspeed.py , DeepSpeedOptimizerWrapper.step is an "pass" function.
                    optimizer.step() 
                    # If use deepseed,`accelerator.backward(loss)` is doing that automatically. Therefore, this function will not work.
                    # For detail, see https://github.com/huggingface/accelerate/blob/main/src/accelerate/utils/deepspeed.py , DeepSpeedOptimizerWrapper.zero_grad is an "pass" function.
                    optimizer.zero_grad() 
                    
                    if accelerator.is_local_main_process:
                        current_loss = loss.item()
                        loss_history.append(current_loss)
                        epoch_loss_sum += current_loss
                        epoch_steps += 1
                        
                        avg_loss_so_far = epoch_loss_sum / epoch_steps
                        global_samples_trained = steps * accelerator.num_processes
                        train_pbar.set_postfix({
                            'loss': f'{current_loss:.4f}',
                            'avg': f'{avg_loss_so_far:.4f}'
                        })
            
            if accelerator.is_local_main_process:
                train_pbar.close()
            
            accelerator.wait_for_everyone()
            
            if accelerator.is_local_main_process:
                logger.info("="*80)
                if use_rag:
                    logger.info("📊 begin evaluation with dynamic context")
                else:
                    logger.info("📊 begin evaluation without dynamic context")
                logger.info("="*80)
            
            model.eval()
            if use_rag:
                eval_loss = evaluate_with_dynamic_context(
                    model=model,
                    processor=processor,
                    eval_dataset=eval_dataset,
                    device=device,
                    accelerator=accelerator,
                    similarity_threshold=0.85,
                    max_lookback=8,
                    top_k=5
                )
            else:
                eval_loss = evaluate(model, eval_loader)
            model.train()
            accelerator.wait_for_everyone()
            
            eval_losses.append(eval_loss)

            # Calculate average loss for this epoch
            if accelerator.is_local_main_process and epoch_steps > 0:
                avg_epoch_loss = epoch_loss_sum / epoch_steps
                epoch_losses.append(avg_epoch_loss)
                
                logger.info("=" * 80)
                logger.info(f"📈 Epoch {epoch+1}/{epochs} done")
                logger.info(f"   training average Loss: {avg_epoch_loss:.6f} ({epoch_steps} batches)")
                logger.info(f"   evaluating average Loss: {eval_loss:.6f}")
                logger.info("=" * 80)

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
                        logger.warning("No LoRA params found when saving. Falling back to full save.")
                    unwrapped_model.save_pretrained(
                        epoch_ckpt_dir,
                        is_main_process=accelerator.is_main_process,
                        save_function=accelerator.save,
                    )
                else:
                    if accelerator.is_local_main_process:
                        logger.info(f"Gathering {len(lora_params)} LoRA param shards with deepspeed...")
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

            action = early_stopper.step(eval_loss)

            # if action == "save":
            #     if accelerator.is_local_main_process:
            #         logger.info("New best model found. Saving checkpoint...")
            #     accelerator.wait_for_everyone()
            #     unwrapped_model = accelerator.unwrap_model(model)
            #     unwrapped_model.save_pretrained(
            #         output_dir + "/best_checkpoint",
            #         is_main_process=accelerator.is_main_process,
            #         save_function=accelerator.save,
            #     )

            if action == "stop":
                if accelerator.is_local_main_process:
                    logger.info(f"Early stopping at epoch {epoch+1}. Best loss = {early_stopper.best_loss}")
                break
                # =================================

    finally:
        if accelerator.is_local_main_process and len(loss_history) > 0:
            logger.info("Plotting loss curves...")
            training_type = "LoRA" if use_lora else "Full"
            plot_training_loss(loss_history, epoch_losses, output_dir, training_type, eval_losses=eval_losses)

    # Synchronize all processes to ensure training completion before saving the model.
    accelerator.wait_for_everyone()
    
    if accelerator.is_local_main_process:
        logger.info("Training completed.")
        logger.info(f"Best model saved at: {output_dir}/best_checkpoint")
        processor.save_pretrained(output_dir)
        write_chat_template(processor, output_dir)

    return epoch_ckpt_dir

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train Qwen2.5-VL model with LoRA or full fine-tuning')
    parser.add_argument('--use_lora', type=bool, default=True, help='Use LoRA (True) or full fine-tuning (False). Default: True')
    parser.add_argument('--use_rag', type=bool, default=True, help='Use RAG evaluation logic (True) or simple evaluation (False). Default: True')
    parser.add_argument('--train-data-path', type=str, required=True, help='Training dataset json path')
    parser.add_argument('--eval-data-path', type=str, required=True, help='Eval dataset json path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for history augmentation')

    args = parser.parse_args()
    train(
        train_data_path=args.train_data_path,
        eval_data_path=args.eval_data_path,
        use_lora=args.use_lora,
        use_rag=args.use_rag,
        seed=args.seed,
    )