import sys
import os
sys.path.append('..')

from transformers import TrainingArguments, Trainer
import torch
from loop_llama_config import LoopLlamaConfig
from loop_llama_model import LoopLlamaForCausalLM
from transformers import AutoTokenizer , AutoModelForCausalLM
from transformers import LlamaConfig
from datasets import load_dataset
from accelerate import Accelerator

## 参数设置
## 数据参数
data_name_or_path = "EleutherAI/the_pile_deduplicated"
max_length = 1024
## 训练参数
per_device_batch_size = 1
num_epochs = 3
learning_rate = 2e-4

## 循环设置
loop_layers = [(6,8)]
loop_strategy = "fixed_count"
loop_count = [3]
kv_cache_mode = "virtual_layers"
virtual_layer_count = [3]
virtual_attention_mode = "parallel"

## 模型设置
original_llama_model_name = "meta-llama/Llama-3.2-1B"
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
  
def preprocess_function(examples, tokenizer):
    """将数据集中的question和answer转换为模型可接受的格式"""
    texts = examples["text"]
    tokenized_inputs = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length
    )
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
    return tokenized_inputs

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
    
    # 预处理数据集
    tokenized_dataset = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    # 训练参数 - Accelerate会自动处理多GPU
    training_args = TrainingArguments(
        output_dir=f"./my_loop_llama_output_f{loop_layers[0][0]}_{loop_layers[0][1]}_{loop_count[0]}",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=8,
        warmup_steps=1,
        learning_rate=learning_rate,
        report_to="tensorboard",
        logging_steps=10,
        save_strategy="epoch",
        logging_dir=f'./logs_f{loop_layers[0][0]}_{loop_layers[0][1]}_{loop_count[0]}',
        eval_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        dataloader_pin_memory=False,  # 多GPU训练时建议设为False
        remove_unused_columns=False
    )
    
    trainer = Trainer(
        model=loop_llama_model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"]
    )
    
    trainer.train()
    return trainer
    
if __name__ == "__main__":
    ## 加载模型
    loop_llama_model = LoopLlamaForCausalLM.from_pretrained(original_llama_model_name, config=config)
    origin_llama_model = AutoModelForCausalLM.from_pretrained(original_llama_model_name)
    tokenizer = AutoTokenizer.from_pretrained(original_llama_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    loop_llama_model.model.training_mode = True
    
    ## 加载数据
    dataset = load_dataset(data_name_or_path)
    
    # 开始训练
    CPT_train(loop_llama_model, dataset, tokenizer) 