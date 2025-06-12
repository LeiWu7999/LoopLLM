from transformers.models.llama.configuration_llama import LlamaConfig
from typing import Optional, Tuple, Union


class LoopLlamaConfig(LlamaConfig):
    """
    扩展LlamaConfig以支持循环层实验配置
    """
    
    model_type = "loop_llama"
    
    def __init__(
        self,
        # 循环层配置
        loop_layers: Optional[Tuple[int, int]] = None,  # (start_layer_idx, end_layer_idx)
        loop_strategy: str = "fixed_count",  # "fixed_count" or "dynamic_stop"
        loop_count: int = 3,  # 固定循环次数
        # 动态停止配置
        cosine_threshold: float = 0.95,  # 余弦相似度阈值
        kl_threshold: float = 0.01,  # KL散度阈值
        max_loop_count: int = 20,  # 最大循环次数（避免无限循环）
        
        # KV缓存维护机制选择
        kv_cache_mode: str = "virtual_layers",  # "virtual_layers" or "merge_strategy"
        
        # 虚拟层映射模式参数 (m->n)
        virtual_layer_count: int = 5,  # 每个物理层对应多少个虚拟层
        min_loop_count: int = 5,  # 最小循环次数
        
        # 虚拟层注意力计算模式
        virtual_attention_mode: str = "parallel",  # "parallel" or "serial"
        
        # 合并策略模式参数 (m->1)
        merge_strategy: str = "ema",  # "ema", "average", "last"
        merge_ema_alpha: float = 0.7,  # EMA参数
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        self.loop_layers = loop_layers
        self.loop_strategy = loop_strategy
        self.loop_count = loop_count
        self.cosine_threshold = cosine_threshold
        self.kl_threshold = kl_threshold
        self.max_loop_count = max_loop_count
        
        # KV缓存配置
        self.kv_cache_mode = kv_cache_mode
        self.virtual_layer_count = virtual_layer_count
        self.min_loop_count = min_loop_count
        self.virtual_attention_mode = virtual_attention_mode
        self.merge_strategy = merge_strategy
        self.merge_ema_alpha = merge_ema_alpha
        
        # 验证配置
        self._validate_loop_config()
    
    def _validate_loop_config(self):
        """验证循环层配置的有效性"""
        if self.loop_layers is not None:
            start_idx, end_idx = self.loop_layers
            if start_idx < 0 or end_idx >= self.num_hidden_layers or start_idx > end_idx:
                raise ValueError(
                    f"无效的循环层配置: loop_layers={self.loop_layers}, "
                    f"模型共有{self.num_hidden_layers}层"
                )
        
        if self.loop_strategy not in ["fixed_count", "dynamic_stop"]:
            raise ValueError(f"不支持的循环策略: {self.loop_strategy}")
        
        if self.kv_cache_mode not in ["virtual_layers", "merge_strategy"]:
            raise ValueError(f"不支持的KV缓存模式: {self.kv_cache_mode}")
        
        if self.kv_cache_mode == "virtual_layers":
            if self.virtual_layer_count <= 0:
                raise ValueError(f"虚拟层数量必须大于0: {self.virtual_layer_count}")
            if self.min_loop_count < self.virtual_layer_count:
                raise ValueError(f"最小循环次数必须大于等于虚拟层数量: {self.min_loop_count} < {self.virtual_layer_count}")
            if self.virtual_attention_mode not in ["parallel", "serial"]:
                raise ValueError(f"不支持的虚拟层注意力模式: {self.virtual_attention_mode}")
            if self.max_loop_count < self.min_loop_count:
                raise ValueError(f"最大循环次数必须大于等于最小循环次数: {self.max_loop_count} < {self.min_loop_count}")
        
        if self.kv_cache_mode == "merge_strategy":
            if self.merge_strategy not in ["ema", "average", "last"]:
                raise ValueError(f"不支持的合并策略: {self.merge_strategy}")
            if self.merge_strategy == "ema" and not (0 < self.merge_ema_alpha < 1):
                raise ValueError(f"EMA参数必须在(0,1)范围内: {self.merge_ema_alpha}") 