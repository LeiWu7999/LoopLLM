import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
import torch
from loop_llama_config import LoopLlamaConfig
from loop_llama_model import LoopLlamaForCausalLM
from transformers import AutoTokenizer , AutoModelForCausalLM
from transformers import LlamaConfig
from datasets import load_from_disk
from accelerate import Accelerator
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime

current_time = datetime.now()
# print(f"当前时间: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")

## 参数设置
## 数据参数
data_name_or_path = "./packed_data/"
max_length = 2048 # This should match the SEQ_LENGTH in tokenize_and_pack.py
## 训练参数
per_device_batch_size = 1
num_epochs = 3
learning_rate = 2e-4
gradient_accumulation_steps = 16 # 确保和deepspeed一致
## 循环设置
loop_layers = [(6,8)]
loop_strategy = "fixed_count"
loop_count = [3]
kv_cache_mode = "virtual_layers"
virtual_layer_count = [3]
virtual_attention_mode = "parallel"

## 模型设置
original_llama_model_name = "mistralai/Mistral-7B-v0.1" 
llama_config = LlamaConfig.from_pretrained(original_llama_model_name)
config_dict = llama_config.to_dict()
config = LoopLlamaConfig(
        loop_layers=loop_layers, 
        loop_strategy=loop_strategy,
        loop_count=loop_count,
        kv_cache_mode=kv_cache_mode,
        virtual_layer_count=virtual_layer_count,
        virtual_attention_mode=virtual_attention_mode,
        **config_dict
    )

@dataclass
class DataCollatorForCrossDocumentAttention:
    """
    Data collator that creates a custom attention mask to prevent attention
    between different documents within a single packed sequence.
    """
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Standard collation to batch tensors
        input_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
        document_ids = torch.tensor([f["document_ids"] for f in features], dtype=torch.long)
        
        batch_size, seq_length = input_ids.shape
        
        # Create the attention mask
        attention_mask = torch.zeros((batch_size, seq_length, seq_length), dtype=torch.float32)

        for i in range(batch_size):
            # 1. Create a causal mask (lower triangular)
            causal_mask = torch.tril(torch.ones((seq_length, seq_length), dtype=torch.bool))
            
            # 2. Create a document boundary mask
            doc_ids = document_ids[i]
            # `doc_ids_matrix[j, k] = 1` if token j and k are from the same document
            doc_ids_matrix = doc_ids.unsqueeze(1) == doc_ids.unsqueeze(0)
            
            # 3. Combine them. The final mask is 1 only if both conditions are met.
            combined_mask = causal_mask & doc_ids_matrix
            attention_mask[i] = combined_mask.float()

        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(), # In LLM pre-training, labels are the input_ids
            "attention_mask": attention_mask
        }
  
def CPT_train(loop_llama_model, dataset, tokenizer, freeze=False):
    if freeze:
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
    data_collator = DataCollatorForCrossDocumentAttention()

    # 训练参数 - Accelerate会自动处理多GPU
    training_args = TrainingArguments(
        output_dir=f"./my_loop_llama_output_f{loop_layers[0][0]}_{loop_layers[0][1]}_{loop_count[0]}",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=1,
        learning_rate=learning_rate,
        report_to="tensorboard",
        logging_steps=10,
        save_strategy="epoch",
        logging_dir=f'./logs_{current_time.strftime("%Y%m%d_%Hh%Mm%Ss")}',
        eval_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        dataloader_pin_memory=False,  # 多GPU训练时建议设为False
        remove_unused_columns=False
    )
    
    trainer = Trainer(
        model=loop_llama_model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"], # MODIFIED: Use the correct split name
        data_collator=data_collator # ADDED: Use our custom collator
    )
    
    trainer.train()
    return trainer
    
if __name__ == "__main__":
    ## 加载模型
    loop_llama_model = LoopLlamaForCausalLM.from_pretrained(original_llama_model_name, config=config)
    # origin_llama_model = AutoModelForCausalLM.from_pretrained(original_llama_model_name)
    tokenizer = AutoTokenizer.from_pretrained(original_llama_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    loop_llama_model.model.training_mode = True
    ## 加载数据
    dataset = load_from_disk(data_name_or_path)
    
    # Split the dataset if it doesn't have train/validation splits
    if "train" not in dataset:
        print("Dataset does not contain train/validation splits. Assuming single split and creating them.")
        # This is a fallback. The packing script should create train/validation folders.
        dataset = dataset.train_test_split(test_size=0.01, seed=42)

    # 开始训练
    CPT_train(loop_llama_model, dataset, tokenizer) 