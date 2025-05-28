# LoopLlama: 循环层增强的LLaMA模型

## 📖 项目简介

LoopLlama是一个基于LLaMA架构的创新语言模型，通过引入**循环层机制**来增强模型的表达能力和推理深度。该项目实现了多种循环策略和KV缓存管理方案，为深度学习研究提供了新的探索方向。

## 🏗️ 核心架构

### 核心文件结构
```
loop_model/
├── loop_llama_config.py    # 配置类，定义所有循环相关参数
├── loop_llama_model.py     # 主模型实现，包含循环层逻辑
├── loop_cache_utils.py     # KV缓存管理，支持多种缓存策略
├── requirements.txt        # 依赖包列表
└── README.md              # 本使用手册
```

### 循环层机制
- **循环层范围**: 指定模型中哪些层参与循环计算
- **循环策略**: 固定次数循环 vs 动态收敛停止
- **KV缓存模式**: 虚拟层映射 vs 合并策略

## 🚀 快速开始

### 1. 环境安装
```bash
# 激活虚拟环境
source loop_env/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 基础使用示例
```python
import torch
from loop_llama_config import LoopLlamaConfig
from loop_llama_model import LoopLlamaForCausalLM

# 创建配置
config = LoopLlamaConfig(
    vocab_size=32000,
    hidden_size=512,
    num_hidden_layers=12,
    num_attention_heads=8,
    loop_layers=(4, 7),  # 第4-7层为循环层
    loop_strategy="fixed_count",
    loop_count=3,
    kv_cache_mode="virtual_layers"
)

# 初始化模型
model = LoopLlamaForCausalLM(config)

# 推理
input_ids = torch.tensor([[1, 2, 3, 4, 5]])
outputs = model(input_ids=input_ids)
logits = outputs.logits
```

## ⚙️ 配置参数详解

### 基础模型参数
```python
config = LoopLlamaConfig(
    vocab_size=32000,           # 词汇表大小
    hidden_size=512,            # 隐藏层维度
    intermediate_size=1024,     # FFN中间层维度
    num_hidden_layers=12,       # 总层数
    num_attention_heads=8,      # 注意力头数
    max_position_embeddings=2048, # 最大位置编码长度
)
```

### 循环层配置
```python
# 循环层范围
loop_layers=(4, 7),  # 第4-7层为循环层，None表示无循环层

# 循环策略
loop_strategy="fixed_count",     # "fixed_count" | "dynamic_stop"
loop_count=3,                    # 固定循环次数
max_loop_count=10,               # 最大循环次数（防止无限循环）

# 动态停止参数
cosine_threshold=0.95,           # 余弦相似度收敛阈值
kl_threshold=0.01,               # KL散度收敛阈值
```

### KV缓存模式
```python
# 模式选择
kv_cache_mode="virtual_layers",  # "virtual_layers" | "merge_strategy"

# 虚拟层模式参数
virtual_layer_count=5,           # 每个物理层对应的虚拟层数
min_loop_count=5,                # 最小循环次数
virtual_attention_mode="parallel", # "parallel" | "serial"

# 合并策略模式参数
merge_strategy="ema",            # "ema" | "average" | "last"
merge_ema_alpha=0.7,             # EMA衰减系数
```

## 🎯 使用场景

### 1. 文本生成
```python
# 配置生成模型
config = LoopLlamaConfig(
    vocab_size=32000,
    hidden_size=512,
    num_hidden_layers=12,
    num_attention_heads=8,
    loop_layers=(4, 7),
    loop_strategy="dynamic_stop",
    cosine_threshold=0.95,
    kv_cache_mode="virtual_layers",
    virtual_attention_mode="parallel"
)

model = LoopLlamaForCausalLM(config)

# 生成文本
input_ids = torch.tensor([[1, 2, 3]])  # 起始token
generated = model.generate(
    input_ids=input_ids,
    max_length=50,
    do_sample=True,
    temperature=0.8
)
```

### 2. 困惑度评估
```python
def calculate_perplexity(model, input_ids):
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss
        ppl = torch.exp(loss)
    return ppl.item()

# 评估模型
test_sequences = [
    torch.tensor([[1, 2, 3, 4, 5]]),
    torch.tensor([[10, 20, 30, 40]])
]

for seq in test_sequences:
    ppl = calculate_perplexity(model, seq)
    print(f"序列 {seq.squeeze().tolist()} 的PPL: {ppl:.4f}")
```

### 3. 批量处理
```python
# 批量推理（推荐用于评估）
batch_input_ids = torch.tensor([
    [1, 2, 3, 4, 5, 0, 0],  # padding到统一长度
    [10, 20, 30, 40, 50, 60, 70]
])
attention_mask = torch.tensor([
    [1, 1, 1, 1, 1, 0, 0],  # 标记有效token
    [1, 1, 1, 1, 1, 1, 1]
])

outputs = model(
    input_ids=batch_input_ids,
    attention_mask=attention_mask,
    labels=batch_input_ids
)
batch_loss = outputs.loss
```

## 🔧 高级功能

### 1. 自定义循环策略
```python
# 动态停止策略
config = LoopLlamaConfig(
    loop_strategy="dynamic_stop",
    cosine_threshold=0.98,      # 更严格的收敛条件
    max_loop_count=15,          # 允许更多循环
    kv_cache_mode="merge_strategy",
    merge_strategy="ema",
    merge_ema_alpha=0.8
)
```

### 2. 不同注意力模式对比
```python
# 并行注意力：所有虚拟层的KV状态拼接
config_parallel = LoopLlamaConfig(
    kv_cache_mode="virtual_layers",
    virtual_attention_mode="parallel",
    virtual_layer_count=3
)

# 串行注意力：只使用当前虚拟层的KV状态
config_serial = LoopLlamaConfig(
    kv_cache_mode="virtual_layers",
    virtual_attention_mode="serial",
    virtual_layer_count=3
)
```

### 3. 内存优化
```python
# 对于大模型，使用合并策略减少内存占用
config = LoopLlamaConfig(
    kv_cache_mode="merge_strategy",
    merge_strategy="last",      # 只保留最后一次循环结果
    loop_count=5
)
```

## 📊 性能监控

### 1. 循环次数统计
```python
# 对于动态停止策略，可以监控实际循环次数
def monitor_loop_steps(model, input_ids):
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=True)
        if hasattr(outputs.past_key_values, 'current_forward_loop_step'):
            loop_steps = outputs.past_key_values.current_forward_loop_step
            print(f"实际循环次数: {loop_steps}")
    return outputs
```

### 2. 内存使用监控
```python
import torch

def monitor_memory():
    if torch.cuda.is_available():
        print(f"GPU内存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU内存缓存: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
```

## ⚠️ 注意事项

### 1. 配置兼容性
- `virtual_layer_count` 必须 ≤ `min_loop_count`
- `max_loop_count` 必须 ≥ `min_loop_count`
- 循环层范围不能超出模型总层数

### 2. 内存管理
- 虚拟层并行模式内存占用较大，适合小模型或充足内存环境
- 合并策略模式内存友好，适合大模型部署
- 长序列推理时建议使用较小的`virtual_layer_count`

### 3. 性能权衡
- 循环次数越多，计算开销越大，但可能获得更好的表示能力
- 动态停止策略可以自适应调整计算量，但增加了判断开销
- 不同KV缓存模式适用于不同的应用场景

## 🔬 实验建议

### 1. 模型对比实验
```python
# 创建对照组：无循环层的基准模型
baseline_config = LoopLlamaConfig(loop_layers=None)
baseline_model = LoopLlamaForCausalLM(baseline_config)

# 实验组：不同循环配置
loop_configs = [
    {"loop_count": 3, "kv_cache_mode": "virtual_layers"},
    {"loop_count": 5, "kv_cache_mode": "merge_strategy"},
    {"loop_strategy": "dynamic_stop", "cosine_threshold": 0.95}
]
```

### 2. 评估指标
- **困惑度(PPL)**: 语言建模能力
- **生成质量**: BLEU、ROUGE等指标
- **计算效率**: 推理时间、内存占用
- **收敛性**: 动态停止策略的循环次数分布

## 📚 扩展开发

### 1. 自定义收敛条件
可以在`loop_llama_model.py`的`_check_convergence`方法中添加新的收敛判断逻辑。

### 2. 新的合并策略
可以在`loop_cache_utils.py`的`_merge_current_forward_history`方法中实现新的KV状态合并算法。

### 3. 循环层选择策略
可以实现动态选择哪些层参与循环的机制，而不是固定的层范围。

## 🤝 贡献指南

1. 保持代码风格一致
2. 添加充分的注释和文档
3. 确保新功能有相应的配置参数
4. 验证功能的正确性和性能影响

## 📄 许可证

本项目遵循MIT许可证，详见LICENSE文件。

---

**LoopLlama** - 探索循环机制在大语言模型中的无限可能 🚀 