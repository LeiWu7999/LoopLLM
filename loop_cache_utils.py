import torch
from typing import Dict, List, Optional, Tuple, Any
from transformers.cache_utils import Cache
import math


class LoopCache(Cache):
    """
    循环层专用的KV缓存管理类
    支持两种维护机制：
    1. 虚拟层映射 (m->n): 将循环层展开为固定数量的虚拟层
    2. 合并策略 (m->1): 将多次循环的KV状态合并为单一缓存
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.loop_start, self.loop_end = config.loop_layers
        self.kv_cache_mode = config.kv_cache_mode
        
        # 普通层的KV缓存
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        
        # 根据模式初始化循环层缓存
        if self.kv_cache_mode == "virtual_layers":
            self._init_virtual_layers_mode()
        elif self.kv_cache_mode == "merge_strategy":
            self._init_merge_strategy_mode()
        else:
            raise ValueError(f"不支持的KV缓存模式: {self.kv_cache_mode}")
        
        self._seen_tokens = 0
        self.current_forward_loop_step = 0  # 当前forward中的循环步数
    
    def _init_virtual_layers_mode(self):
        """初始化虚拟层映射模式"""
        self.virtual_layer_count = self.config.virtual_layer_count
        self.min_loop_count = self.config.min_loop_count
        self.virtual_attention_mode = self.config.virtual_attention_mode
        
        # 为每个循环层创建虚拟层缓存
        # 结构: {layer_idx: {virtual_idx: (key_tensor, value_tensor)}}
        self.virtual_key_cache: Dict[int, Dict[int, torch.Tensor]] = {}
        self.virtual_value_cache: Dict[int, Dict[int, torch.Tensor]] = {}
        
        for layer_idx in range(self.loop_start, self.loop_end + 1):
            self.virtual_key_cache[layer_idx] = {}
            self.virtual_value_cache[layer_idx] = {}
    
    def _init_merge_strategy_mode(self):
        """初始化合并策略模式"""
        self.merge_strategy = self.config.merge_strategy
        self.merge_ema_alpha = self.config.merge_ema_alpha
        
        # 循环层的合并后KV缓存
        self.merged_key_cache: Dict[int, torch.Tensor] = {}
        self.merged_value_cache: Dict[int, torch.Tensor] = {}
        
        # 当前forward中的循环历史（用于合并）
        self.current_forward_key_history: Dict[int, List[torch.Tensor]] = {}
        self.current_forward_value_history: Dict[int, List[torch.Tensor]] = {}
        
        for layer_idx in range(self.loop_start, self.loop_end + 1):
            self.current_forward_key_history[layer_idx] = []
            self.current_forward_value_history[layer_idx] = []
    
    def is_loop_layer(self, layer_idx: int) -> bool:
        """判断是否为循环层"""
        return self.loop_start <= layer_idx <= self.loop_end
    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """更新缓存"""
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
        
        if self.is_loop_layer(layer_idx):
            if self.kv_cache_mode == "virtual_layers":
                return self._update_virtual_layers_cache(key_states, value_states, layer_idx, cache_kwargs)
            else:  # merge_strategy
                return self._update_merge_strategy_cache(key_states, value_states, layer_idx, cache_kwargs)
        else:
            return self._update_normal_cache(key_states, value_states, layer_idx, cache_kwargs)
    
    def _create_empty_cache(self, key_states: torch.Tensor, value_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """创建空的KV缓存"""
        batch_size, num_heads, seq_len, head_dim = key_states.shape
        empty_key = torch.empty(batch_size, num_heads, 0, head_dim, 
                              dtype=key_states.dtype, device=key_states.device)
        empty_value = torch.empty(batch_size, num_heads, 0, head_dim, 
                                dtype=value_states.dtype, device=value_states.device)
        return empty_key, empty_value
    
    def _update_normal_cache(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """更新普通层的缓存"""
        # 确保缓存列表足够长
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(None)
            self.value_cache.append(None)
        
        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
        
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
    
    def _update_virtual_cache_tensor(self, cache_dict: Dict[int, torch.Tensor], new_tensor: torch.Tensor, 
                                    layer_idx: int, virtual_idx: int) -> None:
        """更新单个虚拟层缓存张量的通用方法"""
        if virtual_idx not in cache_dict[layer_idx]:
            # 首次添加
            cache_dict[layer_idx][virtual_idx] = new_tensor
        elif self.current_forward_loop_step < self.virtual_layer_count:
            # 第一轮循环,直接拼接
            cache_dict[layer_idx][virtual_idx] = torch.cat([
                cache_dict[layer_idx][virtual_idx], new_tensor
            ], dim=-2)
        else:
            # 第二轮循环及以后,替换最后一个token的KV状态
            current_cache = cache_dict[layer_idx][virtual_idx]
            if current_cache.shape[-2] > 0:
                cache_dict[layer_idx][virtual_idx] = torch.cat([
                    current_cache[:, :, :-1, :], new_tensor
                ], dim=-2)
            else:
                cache_dict[layer_idx][virtual_idx] = new_tensor

    def _update_virtual_layers_cache(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """更新虚拟层映射模式的缓存"""
        virtual_idx = self.current_forward_loop_step % self.virtual_layer_count
        
        # 更新key和value缓存
        self._update_virtual_cache_tensor(self.virtual_key_cache, key_states, layer_idx, virtual_idx)
        self._update_virtual_cache_tensor(self.virtual_value_cache, value_states, layer_idx, virtual_idx)
        
        # 根据attention模式返回相应的缓存
        if self.virtual_attention_mode == "parallel":
            result = self._get_concatenated_virtual_cache(layer_idx)
        else:  # serial
            result = self._get_current_virtual_cache(layer_idx, virtual_idx)
        
        # 处理空缓存情况
        if result[0] is None:
            return self._create_empty_cache(key_states, value_states)
        
        return result
    
    def _get_concatenated_virtual_cache(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取指定层所有虚拟层的拼接缓存"""
        key_parts = []
        value_parts = []
        
        # 按虚拟层顺序拼接（注意：只拼接已存在的虚拟层）
        for virtual_idx in range(self.virtual_layer_count):
            if virtual_idx in self.virtual_key_cache[layer_idx]:
                key_parts.append(self.virtual_key_cache[layer_idx][virtual_idx])
                value_parts.append(self.virtual_value_cache[layer_idx][virtual_idx])
        
        if not key_parts:
            # 如果没有任何虚拟层缓存，返回None让调用方处理
            return None, None
        
        concatenated_key = torch.cat(key_parts, dim=-2)
        concatenated_value = torch.cat(value_parts, dim=-2)
        
        return concatenated_key, concatenated_value
    
    def _get_current_virtual_cache(self, layer_idx: int, virtual_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取当前虚拟层的缓存（串行模式）"""
        if virtual_idx in self.virtual_key_cache[layer_idx]:
            current_key = self.virtual_key_cache[layer_idx][virtual_idx]
            current_value = self.virtual_value_cache[layer_idx][virtual_idx]
            return current_key, current_value
        else:
            # 如果当前虚拟层没有缓存，返回None，让调用方处理
            return None, None
    
    def _update_merge_strategy_cache(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """更新合并策略模式的缓存"""
        # 将当前循环步的KV状态添加到历史记录
        # 注意：这里存储的是当前token在当前循环步的KV状态 List[(b, h, 1, d)]
        self.current_forward_key_history[layer_idx].append(key_states.clone())
        self.current_forward_value_history[layer_idx].append(value_states.clone())
        
        # 返回历史缓存 + 当前forward的所有循环状态
        if layer_idx in self.merged_key_cache:
            # 有历史缓存的情况：返回 [历史缓存,当前token的当前循环状态] 的拼接
            full_key = torch.cat([self.merged_key_cache[layer_idx], key_states], dim=-2)
            full_value = torch.cat([self.merged_value_cache[layer_idx], value_states], dim=-2)
        else:
            # 没有历史缓存，只有当前token的当前循环状态
            full_key = key_states
            full_value = value_states
        
        return full_key, full_value
    
    def start_new_forward(self):
        """开始新的forward,重置当前forward的循环计数"""
        self.current_forward_loop_step = 0
        
        if self.kv_cache_mode == "merge_strategy":
            # 重置当前forward的循环历史
            for layer_idx in range(self.loop_start, self.loop_end + 1):
                self.current_forward_key_history[layer_idx] = []
                self.current_forward_value_history[layer_idx] = []
    
    def finish_current_forward_loops(self):
        """完成当前forward的循环，合并结果到历史缓存"""
        if self.kv_cache_mode == "merge_strategy":
            for layer_idx in range(self.loop_start, self.loop_end + 1):
                if self.current_forward_key_history[layer_idx]:
                    merged_key, merged_value = self._merge_current_forward_history(layer_idx)
                    
                    # 更新历史缓存
                    if layer_idx in self.merged_key_cache:
                        self.merged_key_cache[layer_idx] = torch.cat([
                            self.merged_key_cache[layer_idx], merged_key
                        ], dim=-2)
                        self.merged_value_cache[layer_idx] = torch.cat([
                            self.merged_value_cache[layer_idx], merged_value
                        ], dim=-2)
                    else:
                        self.merged_key_cache[layer_idx] = merged_key
                        self.merged_value_cache[layer_idx] = merged_value
    
    def _merge_current_forward_history(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """合并当前forward的循环历史"""
        key_history = self.current_forward_key_history[layer_idx]
        value_history = self.current_forward_value_history[layer_idx]
        
        if not key_history:
            raise ValueError(f"层{layer_idx}没有当前forward的循环历史")
        
        if self.merge_strategy == "last":
            # 只保留最后一次循环的结果
            return key_history[-1], value_history[-1]
        
        elif self.merge_strategy == "average":
            # 简单平均
            merged_key = torch.stack(key_history, dim=0).mean(dim=0)
            merged_value = torch.stack(value_history, dim=0).mean(dim=0)
            return merged_key, merged_value
        
        elif self.merge_strategy == "ema":
            # 指数移动平均
            merged_key = key_history[0]
            merged_value = value_history[0]
            
            for i in range(1, len(key_history)):
                merged_key = self.merge_ema_alpha * merged_key + (1 - self.merge_ema_alpha) * key_history[i]
                merged_value = self.merge_ema_alpha * merged_value + (1 - self.merge_ema_alpha) * value_history[i]
            
            return merged_key, merged_value
        
        else:
            raise ValueError(f"不支持的合并策略: {self.merge_strategy}")
    
    def increment_loop_step(self):
        """增加循环步数"""
        self.current_forward_loop_step += 1
    
    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """
        返回真实的token序列长度，用于位置编码计算
        循环层的存在不应该影响token的空间位置信息
        """
        # 统一返回真实的token序列长度，而不是缓存的内部长度
        # 这确保了position_ids的正确性，避免干扰RoPE位置编码
        
        if self.is_loop_layer(layer_idx):
            if self.kv_cache_mode == "virtual_layers":
                # 虚拟层模式：返回真实序列长度
                # 可以通过任意一个虚拟层的长度来推断（因为都对应相同的token序列）
                for virtual_idx in range(self.virtual_layer_count):
                    if virtual_idx in self.virtual_key_cache[layer_idx]:
                        return self.virtual_key_cache[layer_idx][virtual_idx].shape[-2]
                return 0
            else:  # merge_strategy
                # 合并策略模式：返回真实序列长度
                # 历史缓存长度 + 当前token数量（1）
                length = 0
                if layer_idx in self.merged_key_cache:
                    length += self.merged_key_cache[layer_idx].shape[-2]
                # 当前forward只处理一个token，所以加1
                if self.current_forward_key_history[layer_idx]:
                    length += 1  # 当前token
                return length
        else:
            # 普通层：直接返回缓存长度
            if len(self.key_cache) <= layer_idx or self.key_cache[layer_idx] is None:
                return 0
            return self.key_cache[layer_idx].shape[-2]
    
    def get_cache_length(self, layer_idx: Optional[int] = 0) -> int:
        """
        返回实际的KV缓存长度，用于attention计算
        这个长度可能与真实序列长度不同，因为循环层会有特殊的缓存策略
        """
        if self.is_loop_layer(layer_idx):
            if self.kv_cache_mode == "virtual_layers":
                if self.virtual_attention_mode == "parallel":
                    # 并行模式：返回所有虚拟层的总长度
                    total_length = 0
                    for virtual_idx in range(self.virtual_layer_count):
                        if virtual_idx in self.virtual_key_cache[layer_idx]:
                            total_length += self.virtual_key_cache[layer_idx][virtual_idx].shape[-2]
                    return total_length
                else:  # serial
                    # 串行模式：返回当前虚拟层的长度
                    current_virtual_idx = self.current_forward_loop_step % self.virtual_layer_count
                    if current_virtual_idx in self.virtual_key_cache[layer_idx]:
                        return self.virtual_key_cache[layer_idx][current_virtual_idx].shape[-2]
                    return 0
            else:  # merge_strategy
                length = 0
                if layer_idx in self.merged_key_cache:
                    length += self.merged_key_cache[layer_idx].shape[-2]
                # 加上当前forward的循环历史长度
                if self.current_forward_key_history[layer_idx]:
                    for key_tensor in self.current_forward_key_history[layer_idx]:
                        length += key_tensor.shape[-2]
                return length
        else:
            if len(self.key_cache) <= layer_idx or self.key_cache[layer_idx] is None:
                return 0
            return self.key_cache[layer_idx].shape[-2]
    
    def get_max_cache_shape(self) -> Optional[int]:
        """返回最大缓存容量"""
        return None  # 动态缓存，无固定上限 