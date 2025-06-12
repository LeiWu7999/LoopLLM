import torch
from loop_llama_config import LoopLlamaConfig
from loop_llama_model import LoopLlamaForCausalLM
from transformers import AutoTokenizer, LlamaConfig

def test_generation():
    """测试 LoopLlama 文本生成"""
    print("=== LoopLlama 文本生成测试 ===")

    # 1. 首先加载基础的 LlamaConfig
    base_llama_config = LlamaConfig.from_pretrained("/root/models/Llama-3.2-1B")

    # 2. 将基础配置转换为字典，并添加/覆盖 LoopLlama 特定的参数
    loop_config_params = base_llama_config.to_dict()

    # 添加或修改 LoopLlama 特定的配置
    loop_config_params["loop_layers"] = (2, 3)  # 第2-3层为循环层 (示例值，根据模型调整)
    loop_config_params["loop_strategy"] = "fixed_count" # "fixed_count" 或 "dynamic_stop"
    loop_config_params["loop_count"] = 5             # loop_strategy="fixed_count" 时使用
    # loop_config_params["cosine_threshold"] = 0.95  # loop_strategy="dynamic_stop" 时使用
    # loop_config_params["max_loop_count"] = 10      # loop_strategy="dynamic_stop" 时使用，或作为通用上限
    loop_config_params["kv_cache_mode"] = "virtual_layers" # "virtual_layers" 或 "merge_strategy"
    loop_config_params["virtual_layer_count"] = 5    # kv_cache_mode="virtual_layers" 时使用
    loop_config_params["min_loop_count"] = 5         # kv_cache_mode="virtual_layers" 时使用
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
    model = LoopLlamaForCausalLM.from_pretrained("/root/models/Llama-3.2-1B", config=config)
    tokenizer = AutoTokenizer.from_pretrained("/root/models/Llama-3.2-1B")
    model.eval()  # 设置为评估模式

    # 4. 准备输入
    input_text = "The future of AI is"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    print(f"输入序列: {tokenizer.decode(input_ids[0])}")

    # 5. 生成文本
    try:
        with torch.no_grad():
            pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            if pad_token_id is None:
                # 对于某些没有预设 pad_token_id 的 tokenizer，可以临时指定一个
                # 但最佳实践是在 tokenizer config 中定义它
                print("Warning: pad_token_id not set in tokenizer. Using eos_token_id as pad_token_id for generation.")
                pad_token_id = tokenizer.eos_token_id 
                if pad_token_id is None and hasattr(config, 'eos_token_id') and config.eos_token_id is not None:
                     pad_token_id = config.eos_token_id
                elif pad_token_id is None:
                    # 如果eos_token_id也没有，作为最后手段，如果词汇表ID已知，取一个特殊ID
                    # 但这很不理想，因为可能与实际token冲突
                    pad_token_id = 0 # 或 config.vocab_size -1 (如果已知)
                    print(f"Warning: Using fallback pad_token_id: {pad_token_id}")

            generated_ids = model.generate(
                input_ids,
                max_length=100,
                pad_token_id=pad_token_id
            )
            # 仅解码新生成的部分
            generated_text = tokenizer.decode(generated_ids[0, input_ids.shape[1]:], skip_special_tokens=True)

        print(f"生成序列: {generated_text}")

    except Exception as e:
        print(f"文本生成过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        print("请确保模型配置正确，并且输入与模型期望一致。")
        print("对于 LoopLlama，可能需要检查循环层和KV缓存的配置，以及 num_hidden_layers 是否正确设置。")

if __name__ == "__main__":
    test_generation() 