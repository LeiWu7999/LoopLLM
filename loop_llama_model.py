import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, Dict, Any
from transformers.models.llama.modeling_llama import (
    LlamaPreTrainedModel,
    LlamaModel,
    LlamaForCausalLM,
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation.utils import GenerationMixin
import math

from loop_llama_config import LoopLlamaConfig
from loop_cache_utils import LoopCache


class LoopLlamaModel(LlamaModel):
    """
    支持循环层的LLaMA模型，注意all_hidden_states返回的是所有层的未经过norm的hidden_states
    """
    config_class = LoopLlamaConfig
    
    def __init__(self, config: LoopLlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.rotary_emb = LlamaRotaryEmbedding(config)
        
        self.gradient_checkpointing = False

        self.training_mode = None
        
        # 循环控制相关属性
        self.loop_strategy = config.loop_strategy
        self.cosine_threshold = config.cosine_threshold
        self.kl_threshold = config.kl_threshold
        
        # 将循环块信息处理成更易于使用的格式
        self.loop_blocks = config.loop_layers if config.loop_layers is not None else []
        self.loop_block_map = {}
        if self.loop_blocks:
            for i, block in enumerate(self.loop_blocks):
                start_idx, end_idx = block
                self.loop_block_map[start_idx] = {
                    "end_idx": end_idx,
                    "loop_count": config.loop_count[i],
                    "max_loop_count": config.max_loop_count[i],
                    "min_loop_count": config.min_loop_count[i],
                }

        # Initialize weights and apply final processing
        self.post_init()
    
    def get_input_embeddings(self):
        return self.embed_tokens
    
    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if self.training_mode:
            use_cache = False
        
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        # 初始化缓存
        if use_cache and past_key_values is None:
            # 如果定义了循环块，则使用LoopCache
            if self.loop_blocks:
                past_key_values = LoopCache(config=self.config)
            else:
                past_key_values = DynamicCache()
        
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)      # [1, seq_len]
        
        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )       # 4维tensor [batch_size, key_value_length]->[batch_size, 1, query_length, key_value_length]
        
        hidden_states = inputs_embeds
        
        # create position embeddings to be shared across the decoder layers
        # 不需要
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        
        # --- 层执行逻辑 ---
        layer_idx = 0
        while layer_idx < self.config.num_hidden_layers:
            # 检查当前层是否为循环块的起点
            
            if layer_idx in self.loop_block_map:
                block_info = self.loop_block_map[layer_idx]
                loop_start = layer_idx
                loop_end = block_info["end_idx"]

                # 执行循环块
                hidden_states, all_hidden_states, all_self_attns = self._execute_loop_layers(
                    hidden_states=hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    all_hidden_states=all_hidden_states,
                    all_self_attns=all_self_attns,
                    # 循环块特定参数
                    loop_start=loop_start,
                    loop_end=loop_end,
                    loop_count=block_info["loop_count"],
                    max_loop_count=block_info["max_loop_count"],
                    min_loop_count_for_block=block_info["min_loop_count"],
                    **kwargs,
                )
                
                # 将层索引快进到循环块之后
                layer_idx = loop_end + 1
            else:
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)
                # 执行单个普通层
                decoder_layer = self.layers[layer_idx]
                
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
                
                hidden_states = layer_outputs[0]
                
                if output_attentions:
                    all_self_attns += (layer_outputs[1],)
                
                # 移动到下一层
                layer_idx += 1
        
        hidden_states = self.norm(hidden_states)
        
        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
    
    def _execute_loop_layers(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        all_hidden_states: Optional[Tuple[torch.Tensor]] = None,
        all_self_attns: Optional[Tuple[torch.Tensor]] = None,
        # 循环块参数
        loop_start: int = 0,
        loop_end: int = 0,
        loop_count: int = 0,
        max_loop_count: int = 0,
        min_loop_count_for_block: int = 0,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        执行循环层的逻辑
        """
        if loop_count <= 0:
            return hidden_states, all_hidden_states, all_self_attns
            
        loop_layers = self.layers[loop_start:loop_end + 1]

        if isinstance(past_key_values, LoopCache):
            past_key_values.start_new_forward()
        
        # 不再重置循环缓存，而是使用跨forward复用的机制
        
        prev_hidden_states = None
        loop_step = 0
        while loop_step < max_loop_count:
            current_hidden = hidden_states

            
            # 通过循环层块
            for relative_idx, decoder_layer in enumerate(loop_layers):
                if output_hidden_states:
                    all_hidden_states += (current_hidden,)
                layer_idx = loop_start + relative_idx   # 没用
                
                layer_outputs = decoder_layer(
                    current_hidden,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
                
                current_hidden = layer_outputs[0]
                if output_attentions and layer_outputs[1] is not None:
                    all_self_attns += (layer_outputs[1],)
            
            # 完成一次循环块后，增加循环步数
            if isinstance(past_key_values, LoopCache):
                past_key_values.increment_loop_step()
            
            loop_step += 1
            hidden_states = current_hidden
            # 检查停止条件
            if self.loop_strategy == "fixed_count":
                if loop_step >= loop_count:
                    break
            elif self.loop_strategy == "dynamic_stop":
                # 对于虚拟层模式，首先检查最小循环次数
                if (isinstance(past_key_values, LoopCache) and 
                    past_key_values.kv_cache_mode == "virtual_layers" and
                    loop_step < min_loop_count_for_block):
                    # 还没达到最小循环次数，强制继续循环
                    pass
                elif prev_hidden_states is not None:
                    # 达到最小循环次数后，才检查收敛条件
                    should_stop = self._check_convergence(prev_hidden_states, current_hidden)
                    if should_stop:
                        break
                
                # 达到最大循环次数也要停止
                if loop_step >= max_loop_count:
                    break
            
            prev_hidden_states = current_hidden.clone()
            
        
        # 完成循环后，对于合并策略模式，需要合并当前forward的结果
        if isinstance(past_key_values, LoopCache):
            past_key_values.finish_current_forward_loops()
        
        return hidden_states, all_hidden_states, all_self_attns
    
    def _check_convergence(self, prev_hidden: torch.Tensor, curr_hidden: torch.Tensor) -> bool:
        """
        检查是否满足收敛条件
        """
        # 计算余弦相似度
        prev_flat = prev_hidden.view(-1, prev_hidden.size(-1))
        curr_flat = curr_hidden.view(-1, curr_hidden.size(-1))
        
        cosine_sim = F.cosine_similarity(prev_flat, curr_flat, dim=-1).mean()
        
        # 如果余弦相似度足够高，说明收敛
        if cosine_sim >= self.cosine_threshold:
            return True
        
        # 计算KL散度（需要通过unembedding层投影到概率空间）
        # 这里简化实现，只使用余弦相似度
        return False


class LoopLlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    """
    支持循环层的LLaMA因果语言模型
    """
    _tied_weights_keys = ["lm_head.weight"]
    config_class = LoopLlamaConfig
    
    def __init__(self, config: LoopLlamaConfig):
        super().__init__(config)
        self.model = LoopLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights and apply final processing
        self.post_init()
    
    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value
    
    def get_output_embeddings(self):
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    
    def set_decoder(self, decoder):
        self.model = decoder
    
    def get_decoder(self):
        return self.model
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )
        
        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        ) 