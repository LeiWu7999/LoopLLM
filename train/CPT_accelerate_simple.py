import sys
import os
import json
import argparse
from datetime import datetime
from tqdm import tqdm
import torch.multiprocessing as mp

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling, PreTrainedTokenizerBase
import torch
from loop_llama_config import LoopLlamaConfig
from loop_llama_model import LoopLlamaForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaConfig
from datasets import load_from_disk, DatasetDict
from accelerate import Accelerator
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def calculate_ppl_loss(model, tokenizer, encodings, device, window_size, stride, is_evaluating=False):
    """
    Calculates Perplexity (PPL) and Negative Log-Likelihood (NLL) for a given model
    using a sliding window approach.
    """
    model.eval()
    seq_len = encodings.input_ids.size(1)
    
    nll_sum = 0.0
    total_loss_tokens = 0
    
    prev_end_loc = 0
    input_ids_cpu = encodings.input_ids.squeeze(0) 

    for begin_loc in tqdm(range(0, seq_len, stride), desc="Calculating PPL/NLL"):
        end_loc = min(begin_loc + window_size, seq_len)
        
        current_input_ids_chunk = input_ids_cpu[begin_loc:end_loc]
        if current_input_ids_chunk.numel() == 0:
            if end_loc == seq_len: break
            continue

        current_input_ids = current_input_ids_chunk.unsqueeze(0).to(device)
        
        trg_len = end_loc - prev_end_loc
        
        if trg_len <= 0:
            if end_loc > prev_end_loc :
                 prev_end_loc = end_loc
            continue

        target_labels = current_input_ids.clone()
        current_chunk_len = current_input_ids.size(1)
        
        num_to_mask_at_start = current_chunk_len - trg_len
        if num_to_mask_at_start < 0: 
            num_to_mask_at_start = 0
        
        # 在 Hugging Face 的框架中，-100 会被损失函数忽略。
        target_labels[:, :num_to_mask_at_start] = -100

        if (target_labels == -100).all().item():
            if end_loc == seq_len:
                 break
            prev_end_loc = end_loc
            continue
            
        with torch.no_grad():
            outputs = model(current_input_ids, labels=target_labels)
            neg_log_likelihood = outputs.loss

        if neg_log_likelihood is not None:
            batch_size = target_labels.size(0)
            num_valid_target_tokens = (target_labels != -100).sum().item()
            
            num_loss_calc_tokens = 0
            if num_valid_target_tokens > batch_size : 
                num_loss_calc_tokens = num_valid_target_tokens - batch_size
            
            if num_loss_calc_tokens > 0 :
                nll_sum += neg_log_likelihood.item() * num_loss_calc_tokens
                total_loss_tokens += num_loss_calc_tokens
        
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
            
    if total_loss_tokens == 0:
        print("\nWarning: No tokens were processed for PPL/NLL calculation (total_loss_tokens is 0).")
        return float('inf'), float('inf')

    avg_nll = nll_sum / total_loss_tokens
    ppl = torch.exp(torch.tensor(avg_nll)).item()
    return ppl, avg_nll

@dataclass
class DynamicLoopCountDataCollatorForCrossDocumentAttention:
    """
    Data collator that dynamically samples a loop count for each batch.
    It also handles cross-document attention masking.
    This version synchronizes the sampled loop count across all distributed workers.
    """
    config: LoopLlamaConfig
    dynamic_sampling_params: dict
    # 需要匹配query的类型（如bfloat16）
    dtype: torch.dtype = torch.float32

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Standard collation to batch tensors
        input_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
        document_ids = torch.tensor([f["document_ids"] for f in features], dtype=torch.long)
        
        batch_size, seq_length = input_ids.shape
        
        # Create the attention mask
        attention_mask = torch.zeros((batch_size, seq_length, seq_length), dtype=self.dtype)

        for i in range(batch_size):
            # 1. Create a causal mask (lower triangular) torch.tril() - 提取矩阵的下三角部分，上三角部分变为False
            causal_mask = torch.tril(torch.ones((seq_length, seq_length), dtype=torch.bool))
            
            # 2. Create a document boundary mask
            doc_ids = document_ids[i]
            # `doc_ids_matrix[j, k] = 1` if token j and k are from the same document
            doc_ids_matrix = doc_ids.unsqueeze(1) == doc_ids.unsqueeze(0)
            
            # 3. Combine them. The final mask is 1 only if both conditions are met.
            combined_mask = causal_mask & doc_ids_matrix
            attention_mask[i] = combined_mask.to(self.dtype)
            
        # Dynamically sample loop count for the batch.
        # This sampling is synchronized across all workers to ensure workload balance.
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            # In a distributed setting, we use broadcast_object_list to sync the Python integer
            # across all ranks. This method is device-agnostic and works for both
            # CUDA GPUs (NCCL backend) and Ascend NPUs (HCCL backend).
            if torch.distributed.get_rank() == 0:
                # Rank 0 samples the loop count
                r_bar = self.dynamic_sampling_params['r_bar']
                sigma = self.dynamic_sampling_params['sigma']
                max_loops = self.dynamic_sampling_params['max_loops']

                mu = np.log(r_bar) - 0.5 * (sigma**2)
                tau = np.random.normal(mu, sigma)
                rate = np.exp(tau)
                sampled_value = np.random.poisson(rate) + 1
                
                # Use a list to hold the value for broadcasting
                sampled_loop_count_container = [min(sampled_value, max_loops)]
            else:
                # Other ranks initialize a placeholder list
                sampled_loop_count_container = [0]
            
            # Broadcast the object list from rank 0 to all other ranks.
            torch.distributed.broadcast_object_list(sampled_loop_count_container, src=0)
            
            # Extract the synchronized value
            sampled_loop_count = sampled_loop_count_container[0]
        else:
            # For single-worker or non-distributed training, sample directly.
            r_bar = self.dynamic_sampling_params['r_bar']
            sigma = self.dynamic_sampling_params['sigma']
            max_loops = self.dynamic_sampling_params['max_loops']

            # Sample from the distribution
            mu = np.log(r_bar) - 0.5 * (sigma**2)
            tau = np.random.normal(mu, sigma)
            rate = np.exp(tau)
            sampled_loop_count = np.random.poisson(rate) + 1
            
            # Apply truncation
            sampled_loop_count = min(sampled_loop_count, max_loops)
        
        # The model's forward pass needs to accept 'loop_count'
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
            "attention_mask": attention_mask,
            "loop_count": sampled_loop_count
        }

class CustomLoggingTrainer(Trainer):
    """
    自定义Trainer，用于在训练日志中记录额外的损失信息（如aux_loss）。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 用于保存训练过程中的辅助损失值
        self._current_train_losses = {}

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        重写compute_loss以捕获并保存辅助损失。
        """
        # 将 Trainer 传递的额外参数（如 num_items_in_batch）注入到模型输入中，
        # 这样模型内部就可以用它来做损失归一化。
        if 'num_items_in_batch' in kwargs:
            inputs['num_items_in_batch'] = kwargs['num_items_in_batch']

        outputs = model(**inputs)
        
        # 仅在训练阶段保存这些损失，避免影响评估逻辑
        if self.is_in_train:
            if outputs.aux_loss is not None:
                self._current_train_losses['train/aux_loss'] = outputs.aux_loss.item()
            if outputs.loss_pre_loop is not None:
                self._current_train_losses['train/loss_pre_loop'] = outputs.loss_pre_loop
            if outputs.loss_post_loop is not None:
                self._current_train_losses['train/loss_post_loop'] = outputs.loss_post_loop.item()
            
        return (outputs.loss, outputs) if return_outputs else outputs.loss

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        """
        重写log方法，将我们保存的辅助损失添加到日志中。
        为了兼容不同版本的`transformers.Trainer`，其`log`方法的签名可能变化（例如，增加了额外的参数），
        我们使用`*args`和`**kwargs`来捕獲所有传递给`log`的参数，以确保前向兼容性。
        """
        if self.is_in_train and self._current_train_losses:
            logs.update(self._current_train_losses)
            # 清空已记录的损失，为下一个logging step做准备
            self._current_train_losses = {}
        
        super().log(logs, *args, **kwargs)
  
def CPT_train(loop_llama_model, dataset, tokenizer, training_config, resume_from_checkpoint=None, freeze=False):
    if freeze and loop_llama_model.config.loop_layers:
        loop_layers = loop_llama_model.config.loop_layers
        for name, param in loop_llama_model.named_parameters():
            if name.split('.')[2] in [str(x) for x in range(loop_layers[0][0], loop_layers[0][1] + 1)]:
                param.requires_grad = True
                print(f"Unfreezing: {name} , {param.requires_grad}")
            else:
                param.requires_grad = False
                print(f"Freezing: {name} , {param.requires_grad}")
    
    print("--------------------------------")
    total_params = sum(p.numel() for p in loop_llama_model.parameters())
    trainable_params = sum(p.numel() for p in loop_llama_model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params} ({100 * trainable_params / total_params:.2f}%)")
    print("--------------------------------")
    
    # Instantiate our custom data collator
    if loop_llama_model.config.use_dynamic_loop_sampling:
        print("Using data collator with dynamic loop count sampling.")
        data_collator = DynamicLoopCountDataCollatorForCrossDocumentAttention(
            config=loop_llama_model.config,
            dynamic_sampling_params=training_config['dynamic_sampling_params'],
            dtype=loop_llama_model.dtype
        )
    else:
        print("Using standard data collator.")
        # This part might need adjustment if you ever use a non-dynamic collator that
        # still needs cross-document attention. For now, assuming it's a standard one.
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    current_time = datetime.now().strftime("%Y%m%d_%Hh%Mm%Ss")
    output_dir = training_config['training_params']['output_dir_template'].format(
        loop_llama_model.config.loop_layers[0][0], 
        loop_llama_model.config.loop_layers[0][1]
    ) if loop_llama_model.config.loop_layers else training_config['training_params']['output_dir_template'].format('None', 'None')
    logging_dir = training_config['training_params']['logging_dir_template'].format(current_time)

    # 训练参数 - Accelerate会自动处理多GPU
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=logging_dir,
        num_train_epochs=training_config['training_params']['num_train_epochs'],
        per_device_train_batch_size=training_config['training_params']['per_device_train_batch_size'],
        gradient_accumulation_steps=training_config['training_params']['gradient_accumulation_steps'],
        learning_rate=training_config['training_params']['learning_rate'],
        warmup_steps=training_config['training_params']['warmup_steps'],
        weight_decay=training_config['training_params']['weight_decay'],
        adam_beta1=training_config['training_params']['adam_beta1'],
        adam_beta2=training_config['training_params']['adam_beta2'],
        adam_epsilon=training_config['training_params']['adam_epsilon'],
        lr_scheduler_type=training_config['training_params']['lr_scheduler_type'],
        report_to=training_config['training_params']['report_to'],
        logging_steps=training_config['training_params']['logging_steps'],
        save_strategy=training_config['training_params']['save_strategy'],
        save_steps=training_config['training_params']['save_steps'],
        eval_strategy=training_config['training_params']['eval_strategy'],
        eval_steps=training_config['training_params']['eval_steps'],
        save_total_limit=training_config['training_params']['save_total_limit'],
        bf16=training_config['training_params']['bf16'],
        dataloader_pin_memory=training_config['training_params']['dataloader_pin_memory'],
        remove_unused_columns=training_config['training_params']['remove_unused_columns'],
        per_device_eval_batch_size=training_config['training_params']['per_device_eval_batch_size'],
        eval_on_start=training_config['training_params']['eval_on_start'],
        # prediction_loss_only=True,
        # eval_accumulation_steps=1,
    )
    
    trainer = CustomLoggingTrainer(
        model=loop_llama_model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"], 
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    return trainer
    
if __name__ == "__main__":
    # Set the multiprocessing start method to 'spawn'.
    # This is a common fix for issues with CUDA and other hardware backends (like Ascend NPUs),
    # as it ensures child processes start with a clean state without inheriting potentially
    # problematic parent process state.
    try:
        mp.set_start_method('spawn', force=True)
        print("Multiprocessing start method successfully set to 'spawn'.")
    except RuntimeError as e:
        # This might happen if the start method has already been set.
        print(f"Could not set start method: {e}. It might already be set, which is fine.")
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the training configuration file.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to a checkpoint to resume training from.")
    args = parser.parse_args()

    # Load configurations
    training_config = load_config(args.config)
    loop_conf = training_config['loop_config']
    if not loop_conf['use_loop']:
        print("Warning: Loop is disabled in the configuration. Training will be performed without loop.")
        loop_conf['loop_layers'] = None
        loop_conf['loop_count'] = None
    model_conf = training_config['model_config']
    data_conf = training_config['data_params']

    # 加载模型
    model_name_or_path = model_conf['model_name_or_path']
    llama_config = LlamaConfig.from_pretrained(model_name_or_path)
    config_dict = llama_config.to_dict()
    
    config = LoopLlamaConfig(
        **loop_conf,
        **config_dict
    )
    
    loop_llama_model = LoopLlamaForCausalLM.from_pretrained(model_name_or_path, config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    loop_llama_model.model.training_mode = True

    # 加载数据
    data_path = data_conf['data_name_or_path']
    train_path = os.path.join(data_path, 'train')
    validation_path = os.path.join(data_path, 'validation')

    dataset = None
    # 尝试将 train 和 validation 子目录作为独立的数据集加载
    if os.path.isdir(train_path) and os.path.isdir(validation_path):
        try:
            print(f"Found 'train' and 'validation' subdirectories. Attempting to load them separately from {data_path}")
            train_dataset = load_from_disk(train_path)
            validation_dataset = load_from_disk(validation_path)
            dataset = DatasetDict({
                "train": train_dataset,
                "validation": validation_dataset
            })
            print("Successfully loaded 'train' and 'validation' splits into a DatasetDict.")
        except Exception as e:
            print(f"Tried to load sub-folders but failed: {e}. Falling back to loading the main directory.")
            dataset = None

    if dataset is None:
        print(f"Loading dataset directly from {data_path}.")
        dataset = load_from_disk(data_path)
    
    # Split the dataset if it doesn't have train/validation splits
    if "train" not in dataset or "validation" not in dataset:
        print("Dataset does not contain 'train' and/or 'validation' splits. Assuming single split and creating them.")
        dataset = dataset.train_test_split(test_size=0.01, seed=42)

    # 开始训练
    CPT_train(loop_llama_model, dataset, tokenizer, training_config, resume_from_checkpoint=args.resume_from_checkpoint) 