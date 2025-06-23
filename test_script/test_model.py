'''测试model.model的forward'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import sys
sys.path.append("/home/ubuntu/Documents/newdisk_22T/twq/recurrent_model/LoopLLM")
from loop_llama_config import LoopLlamaConfig
from loop_llama_model import LoopLlamaForCausalLM
from transformers import AutoTokenizer, LlamaConfig

# 1. 首先加载基础的 LlamaConfig
base_llama_config = LlamaConfig.from_pretrained("meta-llama/Llama-3.2-1B")

# 2. 将基础配置转换为字典，并添加/覆盖 LoopLlama 特定的参数
loop_config_params = base_llama_config.to_dict()

loop_count = 10

# 添加或修改 LoopLlama 特定的配置
loop_config_params["loop_layers"] = [[2, 3]]  # 第2-3层为循环层 (示例值，根据模型调整)
loop_config_params["loop_strategy"] = "fixed_count" # "fixed_count" 或 "dynamic_stop"
loop_config_params["loop_count"] = loop_count             # loop_strategy="fixed_count" 时使用
# loop_config_params["cosine_threshold"] = 0.95  # loop_strategy="dynamic_stop" 时使用
loop_config_params["max_loop_count"] = 100      # loop_strategy="dynamic_stop" 时使用，或作为通用上限
loop_config_params["kv_cache_mode"] = "merge_strategy" # "virtual_layers" 或 "merge_strategy"
# loop_config_params["virtual_layer_count"] = loop_count    # kv_cache_mode="virtual_layers" 时使用
# loop_config_params["min_loop_count"] = loop_count         # kv_cache_mode="virtual_layers" 时使用
loop_config_params["virtual_attention_mode"] = "serial" # kv_cache_mode="virtual_layers" 时使用
# loop_config_params["merge_strategy"] = "ema"      # kv_cache_mode="merge_strategy" 时使用
# loop_config_params["merge_ema_alpha"] = 0.7       # merge_strategy="ema" 时使用

# 如果原始 LlamaConfig 中没有 num_hidden_layers，确保它存在，因为 _validate_loop_config 需要它
# 通常 from_pretrained 会包含这个
if "num_hidden_layers" not in loop_config_params:
    # 这个值需要与你的模型结构匹配
    loop_config_params["num_hidden_layers"] = base_llama_config.num_hidden_layers # 或从基础配置中获取

config = LoopLlamaConfig(**loop_config_params)

# 3. 初始化模型和 Tokenizer
model = LoopLlamaForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", config=config, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model.eval()  # 设置为评估模式

# 4. 准备输入并进行生成测试
prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt")

# 使用 model.generate() 进行测试
# 将输入移动到模型所在的设备
inputs = {k: v.to(model.device) for k, v in inputs.items()}


# 5. (可选) 使用 model.model 进行前向传播测试
print("\nTesting forward pass with model.model:")
with torch.no_grad():
    # model.model 是底层的 transformer 模型
    
    outputs = model.model(
        input_ids=inputs["input_ids"],
        # attention_mask=inputs["attention_mask"],  # 移除 attention_mask，让模型自动处理
    )
    # 获取模型输出的logits并转换为token id
    outputs_logits = model.lm_head(outputs.last_hidden_state)
    next_token_id = torch.argmax(outputs_logits[:, -1:, :], dim=-1)
    
    # 将新生成的token添加到输入中
    inputs['input_ids'] = torch.cat([inputs['input_ids'], next_token_id], dim=1)

    for i in range(100):
        outputs = model.model(
            input_ids=next_token_id,
            past_key_values=outputs.past_key_values,
        )
        outputs_logits = model.lm_head(outputs.last_hidden_state)
        next_token_id = torch.argmax(outputs_logits[:, -1:, :], dim=-1)
        print(f'kv_seq_len: {outputs.past_key_values.get_seq_length()}')
    outputs_ids = torch.cat([inputs['input_ids'], next_token_id], dim=1)
    output_text = tokenizer.decode(outputs_ids[0], skip_special_tokens=True)
    print(output_text)

