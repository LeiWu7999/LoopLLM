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
from transformers.utils import ModelOutput
from dataclasses import dataclass
import math

from loop_llama_config import LoopLlamaConfig
from loop_cache_utils import LoopCache

@dataclass
class LoopModelOutputWithPast(BaseModelOutputWithPast):
    """
    继承自BaseModelOutputWithPast，额外包含one_loop_hidden用于辅助损失计算
    """
    # 新增字段：用于辅助损失计算的one_loop_hidden
    one_loop_hidden: Optional[torch.FloatTensor] = None


@dataclass
class LoopCausalLMOutputWithPast(CausalLMOutputWithPast):
    """
    继承自CausalLMOutputWithPast，额外包含辅助损失信息
    """
    # 新增字段：辅助损失相关信息
    aux_loss: Optional[torch.FloatTensor] = None
    loss_pre_loop: Optional[torch.FloatTensor] = None
    loss_post_loop: Optional[torch.FloatTensor] = None

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
        loop_count: Optional[int] = None,
        **kwargs,
    ) -> LoopModelOutputWithPast:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # 训练时根据配置决定是否禁用KV缓存
        # use_kv_cache_in_training=False: 无状态循环
        # use_kv_cache_in_training=True: 有状态循环
        if self.training_mode and not getattr(self.config, 'use_kv_cache_in_training', True):
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
        
        if attention_mask is not None and attention_mask.dim() == 3:
            # Assume 3D attention_mask is [batch_size, seq_len, seq_len]
            # and expand it to [batch_size, 1, seq_len, seq_len]
            attention_mask = attention_mask.unsqueeze(1)
        
        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )       # 4维tensor [batch_size, key_value_length]->[batch_size, 1, query_length, key_value_length]
        
        hidden_states = inputs_embeds
        
        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        
        # --- 层执行逻辑 ---
        layer_idx = 0
        if self.training:
            one_loop_hidden = hidden_states.clone().detach()
        else:
            one_loop_hidden = None
        while layer_idx < self.config.num_hidden_layers:
            # 检查当前层是否为循环块的起点
            if layer_idx in self.loop_block_map:
                block_info = self.loop_block_map[layer_idx]
                loop_start = layer_idx
                loop_end = block_info["end_idx"]
                current_loop_count = loop_count if loop_count is not None else block_info["loop_count"]

                # 执行循环块
                hidden_states, all_hidden_states, all_self_attns = self._execute_loop_layers(
                    hidden_states=hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    cache_position=cache_position,
                    all_hidden_states=all_hidden_states,
                    all_self_attns=all_self_attns,
                    loop_start=loop_start,
                    loop_end=loop_end,
                    loop_count=current_loop_count,
                    max_loop_count=block_info["max_loop_count"],
                    min_loop_count_for_block=block_info["min_loop_count"],
                )
                if one_loop_hidden is not None:
                    with torch.no_grad():
                        one_loop_hidden, _, _ = self._execute_loop_layers(
                        hidden_states=one_loop_hidden,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        use_cache=False,
                        position_embeddings=position_embeddings,
                        output_attentions=False,
                        output_hidden_states=False,
                        # 循环块特定参数
                        loop_start=loop_start,
                        loop_end=loop_end,
                        loop_count=1,
                        max_loop_count=block_info["max_loop_count"],
                        min_loop_count_for_block=block_info["min_loop_count"],
                        past_key_values=None, # 明确不使用缓存
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
                )
                if one_loop_hidden is not None:
                    with torch.no_grad():
                        layer_outputs_aux = decoder_layer(
                            one_loop_hidden,
                            attention_mask=causal_mask,
                            position_ids=position_ids,
                            past_key_value=None, # 明确不使用缓存
                            output_attentions=False, # 明确不计算attention
                            use_cache=False, # 明确不使用缓存
                            cache_position=cache_position,
                            position_embeddings=position_embeddings,
                        )
                        one_loop_hidden = layer_outputs_aux[0]
                hidden_states = layer_outputs[0]
                
                if output_attentions:
                    all_self_attns += (layer_outputs[1],)
                
                # 移动到下一层
                layer_idx += 1
        
        hidden_states = self.norm(hidden_states)
        if one_loop_hidden is not None:
            one_loop_hidden = self.norm(one_loop_hidden)
        
        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        return LoopModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            one_loop_hidden=one_loop_hidden,
        )
    
    def _execute_loop_layers(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        all_hidden_states: Optional[Tuple[torch.Tensor]] = None,
        all_self_attns: Optional[Tuple[torch.Tensor]] = None,
        # 循环块参数
        loop_start: int = 0,
        loop_end: int = 0,
        loop_count: int = 0,
        max_loop_count: int = 0,
        min_loop_count_for_block: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        执行循环层的逻辑
        """
        if loop_count <= 0:
            # if output_hidden_states:
            #     # When skipping the loop, we treat the layers as identity functions.
            #     # The input to each layer in the block is the same `hidden_states` tensor.
            #     # This is crucial to keep the `all_hidden_states` indexing consistent.
            #     for _ in range(loop_start, loop_end + 1):
            #         all_hidden_states += (hidden_states,)
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
                
                layer_outputs = decoder_layer(
                    current_hidden,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
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
            if self.loop_strategy == "fixed_count" or use_cache == False:
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
        loop_count: Optional[int] = None,
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
            loop_count=loop_count,
            **kwargs,
        )
        
        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        
        loss = None
        aux_loss = None
        loss_pre_loop_val = None
        loss_post_loop = None
        
        if labels is not None:
            # print(kwargs['num_items_in_batch'])
            if self.training:
                # 统一处理 labels, 避免在两个地方重复移动到device
                labels = nn.functional.pad(labels, (0, 1), value=-100)
                shift_labels = labels[..., 1:].contiguous()
                shift_labels = shift_labels.view(-1)
                
                # 步骤1: 首先计算辅助损失中的 loss_pre_loop
                # 目的是先完成这部分的计算，并及时释放内存，从而降低峰值内存占用
                aux_loss = torch.tensor(0.0, device=logits.device)
                loss_pre_loop_val = None # 用于保存loss_pre_loop的标量值

                if outputs.one_loop_hidden is not None:
                    # 使用 no_grad 确保这部分计算不构建计算图
                    with torch.no_grad():
                        # 计算 one_loop_hidden 对应的 logits
                        one_loop_logits = self.lm_head(outputs.one_loop_hidden[:, slice_indices, :])
                        one_loop_logits = one_loop_logits.float()
                        one_loop_logits = one_loop_logits.view(-1, self.config.vocab_size)
                        
                        # 计算循环前的损失 Loss_pre_loop
                        shift_labels_gpu = shift_labels.to(one_loop_logits.device)
                        loss_pre_loop = nn.functional.cross_entropy(one_loop_logits, shift_labels_gpu, ignore_index=-100, reduction="sum")
                        loss_pre_loop = loss_pre_loop / kwargs['num_items_in_batch']
                        loss_pre_loop_val = loss_pre_loop.item()
                    
                    # 步骤2: 手动释放不再需要的张量，清空缓存，进一步降低内存
                    del one_loop_logits, loss_pre_loop, shift_labels_gpu
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # 步骤3: 计算主损失 Loss_post_loop，这部分会构建用于反向传播的计算图
                logits = logits.float()
                logits = logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.to(logits.device)
                
                loss_post_loop = nn.functional.cross_entropy(logits, shift_labels, ignore_index=-100, reduction="sum")
                loss_post_loop = loss_post_loop / kwargs['num_items_in_batch']

                # 步骤4: 使用之前计算出的 loss_pre_loop_val 计算 Hinge Loss
                if loss_pre_loop_val is not None:
                    margin = self.config.aux_loss_margin
                    # 这里 loss_pre_loop_val 是一个标量，等效于 loss_pre_loop.detach()，不会引入额外的计算图
                    aux_loss = torch.clamp(loss_post_loop - loss_pre_loop_val + margin / (kwargs['num_items_in_batch'] / logits.shape[0]), min=0.0)
                
                # 总损失: Loss_total = Loss_post_loop + α * Loss_aux
                alpha = self.config.aux_loss_weight
                loss = loss_post_loop + alpha * aux_loss
                
            else:
                loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
        
        return LoopCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            aux_loss=aux_loss,
            loss_pre_loop=loss_pre_loop_val,
            loss_post_loop=loss_post_loop,
        ) 