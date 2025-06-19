#!/usr/bin/env python3
"""
LoopLlama 基础使用示例
演示如何创建、配置和使用循环层增强的LLaMA模型
"""

import torch
from loop_llama_config import LoopLlamaConfig
from loop_llama_model import LoopLlamaForCausalLM


def example_basic_usage():
    """基础使用示例"""
    print("=== LoopLlama 基础使用示例 ===")
    
    # 1. 创建配置
    config = LoopLlamaConfig(
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=6,
        num_attention_heads=4,
        max_position_embeddings=512,
        loop_layers=(2, 3),  # 第2-3层为循环层
        loop_strategy="fixed_count",
        loop_count=3,
        kv_cache_mode="virtual_layers",
        virtual_layer_count=3,
        virtual_attention_mode="parallel"
    )
    
    # 2. 初始化模型
    model = LoopLlamaForCausalLM(config)
    model.eval()
    
    # 3. 准备输入
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    
    # 4. 模型推理
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits
    
    print(f"输入序列: {input_ids.squeeze().tolist()}")
    print(f"输出logits形状: {logits.shape}")
    print(f"预测的下一个token: {torch.argmax(logits[0, -1, :]).item()}")


def example_different_modes():
    """不同循环模式对比示例"""
    print("\n=== 不同循环模式对比 ===")
    
    base_config = {
        "vocab_size": 1000,
        "hidden_size": 128,
        "num_hidden_layers": 6,
        "num_attention_heads": 4,
        "loop_layers": (2, 3),
        "loop_count": 3
    }
    
    # 测试不同模式
    modes = [
        ("虚拟层-并行", {"kv_cache_mode": "virtual_layers", "virtual_attention_mode": "parallel", "virtual_layer_count": 3}),
        ("虚拟层-串行", {"kv_cache_mode": "virtual_layers", "virtual_attention_mode": "serial", "virtual_layer_count": 3}),
        ("合并策略-EMA", {"kv_cache_mode": "merge_strategy", "merge_strategy": "ema", "merge_ema_alpha": 0.7})
    ]
    
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    
    for mode_name, mode_config in modes:
        config = LoopLlamaConfig(**{**base_config, **mode_config})
        model = LoopLlamaForCausalLM(config)
        model.eval()
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            next_token = torch.argmax(outputs.logits[0, -1, :]).item()
        
        print(f"{mode_name}: 预测下一个token = {next_token}")


def example_perplexity_calculation():
    """困惑度计算示例"""
    print("\n=== 困惑度计算示例 ===")
    
    config = LoopLlamaConfig(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=6,
        num_attention_heads=4,
        loop_layers=(2, 3),
        loop_strategy="fixed_count",
        loop_count=3,
        kv_cache_mode="virtual_layers"
    )
    
    model = LoopLlamaForCausalLM(config)
    model.eval()
    
    # 测试序列
    test_sequences = [
        torch.tensor([[1, 2, 3, 4, 5]]),
        torch.tensor([[10, 20, 30, 40]]),
        torch.tensor([[100, 200, 300, 400, 500, 600]])
    ]
    
    for i, seq in enumerate(test_sequences):
        with torch.no_grad():
            outputs = model(input_ids=seq, labels=seq)
            loss = outputs.loss
            ppl = torch.exp(loss)
        
        print(f"序列 {i+1} {seq.squeeze().tolist()}: PPL = {ppl.item():.4f}")


def example_generation():
    """文本生成示例"""
    print("\n=== 文本生成示例 ===")
    
    config = LoopLlamaConfig(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=6,
        num_attention_heads=4,
        loop_layers=(2, 3),
        loop_strategy="dynamic_stop",
        cosine_threshold=0.95,
        max_loop_count=5,
        kv_cache_mode="virtual_layers"
    )
    
    model = LoopLlamaForCausalLM(config)
    model.eval()
    
    # 起始序列
    input_ids = torch.tensor([[1, 2, 3]])
    print(f"起始序列: {input_ids.squeeze().tolist()}")
    
    # 逐步生成
    current_sequence = input_ids.clone()
    past_key_values = None
    
    for step in range(5):
        with torch.no_grad():
            if step == 0:
                outputs = model(input_ids=current_sequence, past_key_values=None, use_cache=True)
            else:
                outputs = model(input_ids=new_token.unsqueeze(0), past_key_values=past_key_values, use_cache=True)
            
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            new_token = torch.argmax(next_token_logits, dim=-1)
            current_sequence = torch.cat([current_sequence, new_token.unsqueeze(0)], dim=1)
        
        print(f"Step {step+1}: {current_sequence.squeeze().tolist()}")


if __name__ == "__main__":
    # 运行所有示例
    example_basic_usage()
    example_different_modes()
    example_perplexity_calculation()
    example_generation()
    
    print("\n=== 示例完成 ===")
    print("更多详细用法请参考 README.md") 