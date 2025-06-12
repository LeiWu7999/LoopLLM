from transformers.models.llama.configuration_llama import LlamaConfig
from typing import Optional, Tuple, Union, List


class LoopLlamaConfig(LlamaConfig):
    """
    扩展LlamaConfig以支持循环层实验配置
    """
    
    model_type = "loop_llama"
    
    def __init__(
        self,
        # 循环层配置
        loop_layers: Optional[List[Tuple[int, int]]] = None,  # (start_layer_idx, end_layer_idx) 的列表
        loop_strategy: str = "fixed_count",  # "fixed_count" or "dynamic_stop"
        loop_count: Union[int, List[int]] = 5,  # 固定循环次数，可为单个整数或列表
        # 动态停止配置
        cosine_threshold: float = 0.95,  # 余弦相似度阈值
        kl_threshold: float = 0.01,  # KL散度阈值

        max_loop_count: Union[int, List[int]] = 10,  # 最大循环次数，可为单个整数或列表

        
        # KV缓存维护机制选择
        kv_cache_mode: str = "virtual_layers",  # "virtual_layers" or "merge_strategy"
        
        # 虚拟层映射模式参数 (m->n)
        virtual_layer_count: Union[int, List[int]] = 5,  # 每个物理层对应多少个虚拟层
        min_loop_count: Union[int, List[int]] = 5,  # 最小循环次数
        
        # 虚拟层注意力计算模式
        virtual_attention_mode: str = "parallel",  # "parallel" or "serial"
        
        # 合并策略模式参数 (m->1)
        merge_strategy: str = "ema",  # "ema", "average", "last"
        merge_ema_alpha: float = 0.7,  # EMA参数
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        self.loop_layers = sorted(loop_layers, key=lambda x: x[0]) if loop_layers is not None else None
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
            # 如果这些参数是int，则扩展为列表
            self.loop_count = [self.loop_count] * len(self.loop_layers) if isinstance(self.loop_count, int) else self.loop_count
            self.max_loop_count = [self.max_loop_count] * len(self.loop_layers) if isinstance(self.max_loop_count, int) else self.max_loop_count
            self.virtual_layer_count = [self.virtual_layer_count] * len(self.loop_layers) if isinstance(self.virtual_layer_count, int) else self.virtual_layer_count
            self.min_loop_count = [self.min_loop_count] * len(self.loop_layers) if isinstance(self.min_loop_count, int) else self.min_loop_count

            if not isinstance(self.loop_layers, list) or not all(isinstance(i, tuple) and len(i) == 2 for i in self.loop_layers):
                raise ValueError("`loop_layers` must be a list of lists, where each inner list contains two integers.")

            # 检查循环次数配置
            if isinstance(self.loop_count, list) and len(self.loop_count) != len(self.loop_layers):
                raise ValueError("If `loop_count` is a list, its length must match the number of blocks in `loop_layers`.")

            # 检查层索引有效性并检查重叠
            for i, (start_idx, end_idx) in enumerate(self.loop_layers):
                if start_idx < 0 or end_idx >= self.num_hidden_layers or start_idx > end_idx:
                    raise ValueError(
                        f"Invalid loop layer configuration in block {i}: loop_layers=[{start_idx}, {end_idx}], "
                        f"but the model has {self.num_hidden_layers} layers."
                    )
                # 检查与前一个块是否重叠
                if i > 0 and start_idx <= self.loop_layers[i-1][1]:
                    raise ValueError(
                        f"Loop blocks cannot overlap. Block {i-1} {self.loop_layers[i-1]} and block {i} {self.loop_layers[i]} are overlapping."
                    )


            for i in range(len(self.loop_layers)):
                if self.virtual_layer_count[i] <= 0:
                    raise ValueError(f"虚拟层数量必须大于0: {self.virtual_layer_count[i]}")
                if self.min_loop_count[i] < self.virtual_layer_count[i]:
                    raise ValueError(f"Block {i}: min_loop_count ({self.min_loop_count[i]}) must be >= virtual_layer_count ({self.virtual_layer_count[i]}).")
                if self.virtual_attention_mode not in ["parallel", "serial"]:
                    raise ValueError(f"Unsupported virtual attention mode: {self.virtual_attention_mode}")
                if self.max_loop_count[i] < self.min_loop_count[i]:
                    raise ValueError(f"Block {i}: max_loop_count ({self.max_loop_count[i]}) must be >= min_loop_count ({self.min_loop_count[i]}).")
                if self.loop_count[i] != self.virtual_layer_count[i]:
                    raise ValueError(f"Block {i}: In virtual_layers mode, loop_count ({self.loop_count[i]}) must be equal to virtual_layer_count ({self.virtual_layer_count[i]}).")

        if self.loop_strategy not in ["fixed_count", "dynamic_stop"]:
            raise ValueError(f"Unsupported loop strategy: {self.loop_strategy}")
        
        if self.kv_cache_mode not in ["virtual_layers", "merge_strategy"]:
            raise ValueError(f"Unsupported KV cache mode: {self.kv_cache_mode}")
            
        if self.kv_cache_mode == "merge_strategy":
            if self.merge_strategy not in ["ema", "average", "last"]:
                raise ValueError(f"Unsupported merge strategy: {self.merge_strategy}")
            if self.merge_strategy == "ema" and not (0 < self.merge_ema_alpha < 1):
                raise ValueError(f"EMA alpha must be in (0, 1), but got {self.merge_ema_alpha}.") 