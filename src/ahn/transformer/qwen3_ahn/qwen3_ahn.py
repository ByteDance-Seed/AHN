# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import torch
import wandb
import random
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.attention.flex_attention import (
    create_block_mask,
    flex_attention,
)
from typing import Union, Optional, Tuple, List, Dict, Any

from ahn.transformer.qwen3 import (
    Qwen3Config as Qwen3Config_,
    Qwen3Model as Qwen3Model_,
    Qwen3ForCausalLM as Qwen3ForCausalLM_
)
from ahn.transformer.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3MLP,
    Qwen3DecoderLayer,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    repeat_kv,
    apply_rotary_pos_emb,
    rotate_half
)

from ahn.utils import BaseAHN, AHNRouter
from ahn.rnn import DeltaNet, GatedDeltaNet, Mamba2
from dataclasses import dataclass
from fla.models.utils import Cache as MemCache
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_flash_attention_utils import _flash_attention_forward, FlashAttentionKwargs
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import ModelOutput, is_flash_attn_greater_or_equal_2_10

# FlexAttention configuration
torch._dynamo.config.cache_size_limit = 512
torch._dynamo.config.accumulated_cache_size_limit = 4096
flex_attention = torch.compile(flex_attention)

@dataclass
class CustomizedModelOutputWithPast(ModelOutput):
    last_hidden_states: torch.FloatTensor = None
    last_mem_hidden_states: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    mem_past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    all_distill_pairs: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None


@dataclass
class CustomizedCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    mem_past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class Qwen3Config(Qwen3Config_):
    r"""
    Args:
        _layer_implementation (`str`, *optional*, defaults to 'Qwen3DecoderLayer'):
            Decoder layer implementation.
        _ahn_implementation (`str`, *optional*):
            Memory implementation of decoder layers.
    """
    def __init__(
        self,
        loss_type: Optional[str] = "ce",
        _layer_implementation: Optional[str] = "Qwen3DecoderLayer",
        _ahn_implementation: Optional[str] = None,
        sliding_window_type: Optional[str] = "fixed",
        ahn_position: Optional[str] = "prefix",
        fa_layer_ids: Optional[str] = "",
        use_ahn_router: Optional[bool] = False,
        use_q_proj: Optional[bool] = False,
        use_normalized_l2: Optional[bool] = True,
        use_layerwise_decay: Optional[bool] = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.loss_type = (
            loss_type.split("+") if not isinstance(loss_type, list) else loss_type
        )

        for loss_i in self.loss_type:
            assert loss_i in [
                "ce",
                "kl",
                "distill",
            ], f"Unsupported loss type: {loss_i}."

        self._layer_implementation = _layer_implementation
        self._ahn_implementation = _ahn_implementation
        self.sliding_window_type = sliding_window_type
        self.sliding_window = kwargs.get("sliding_window", None)
        self.ahn_position = ahn_position
        self.fa_layer_ids = fa_layer_ids
        self.use_ahn_router = use_ahn_router
        self.use_q_proj = use_q_proj
        self.use_normalized_l2 = use_normalized_l2
        self.use_layerwise_decay = use_layerwise_decay


class Qwen3MemFlexAttn(Qwen3Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    @staticmethod
    def pad_sequence(x: torch.Tensor, pad_size: int) -> torch.Tensor:
        B, L, H, D = x.shape
        pad_tensor = x.new_zeros((B, pad_size, H, D))
        return torch.cat([x, pad_tensor], dim=1)

    def pre_attn_forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, int, int]:
        bsz, q_len, _ = hidden_states.size()
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        assert (
            position_embeddings is not None
        ), "Please provide 'position_embeddings' for model forward pass."
        cos, sin = position_embeddings

        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
            }  # Specific to RoPE models
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        # Reashape to match the expected input shape for FlashAttention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        return (
            query_states,
            key_states,
            value_states,
            dropout_rate,
            bsz,
            q_len,
        )

    def flex_attn_forward(
        self,
        bsz: int,
        q_len: int,
        attention_mask: torch.Tensor,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        padded_lengths: List[int] = [2048, 4096, 8192, 16384, 24576, 32768],
    ):
        # Ensure sequence length is a multiple of 128 for FlexAttention
        padded_seq_len = next((l for l in padded_lengths if l >= q_len), (q_len // 128 + 1) * 128)
        pad_size = padded_seq_len - q_len

        flex_query_states = self.pad_sequence(query_states, pad_size).transpose(1, 2)
        flex_key_states = self.pad_sequence(key_states, pad_size).transpose(1, 2)
        flex_value_states = self.pad_sequence(value_states, pad_size).transpose(1, 2)
        flex_attn_output = flex_attention(
            flex_query_states,  # 'flex_query_states': (B, H, L, D)
            flex_key_states,
            flex_value_states,
            enable_gqa=True,
            block_mask=attention_mask,
        )
        # Remove padding from the output
        end_idx = flex_attn_output.shape[2] - pad_size
        flex_attn_output = (
            flex_attn_output[:, :, :end_idx, :].permute(0, 2, 1, 3).contiguous()
        )

        flex_attn_output = flex_attn_output.reshape(bsz, q_len, -1).contiguous()
        return flex_attn_output

    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_inference(*args, **kwargs)

    def forward_train(
        self,
        enable_ahn: bool,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        (
            query_states,
            key_states,
            value_states,
            dropout_rate,
            bsz,
            q_len,
        ) = self.pre_attn_forward(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

        if enable_ahn:
            attn_output = self.flex_attn_forward(
                bsz=bsz,
                q_len=q_len,
                attention_mask=attention_mask,
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
            )
        else:
            attn_output = _flash_attention_forward(
                query_states,
                key_states,
                value_states,
                attention_mask,
                q_len,
                position_ids=position_ids,
                dropout=dropout_rate,
                sliding_window=None, # Hardcoded here
                is_causal=self.is_causal,
            )
            attn_output = attn_output.reshape(
                bsz, q_len, -1
            ).contiguous()

        # TODO: Consider to delete the following 3 lines.
        attn_weights = None
        if output_attentions:
            attn_weights = None # FlashAttention doesn't return attention weights

        hidden_states = self.o_proj(attn_output)

        if enable_ahn:
            return (
                hidden_states,
                attn_weights,
                attn_output,
                (
                    query_states,
                    key_states,
                    value_states,
                ),
            )

        return hidden_states, attn_weights, attn_output

    def forward_inference(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        num_attn_sinks: int = 128,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        enable_ahn: bool = False,
    ):
        (
            query_states,
            key_states,
            value_states,
            dropout_rate,
            bsz,
            q_len,
        ) = self.pre_attn_forward(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

        if enable_ahn:
            if query_states.shape != key_states.shape:
                # Decoding with FlashAttention
                assert query_states.shape[1] == 1
                tmp_key_states = torch.cat(
                    [
                        key_states[:, :num_attn_sinks],
                        key_states[:, num_attn_sinks + 1 :],
                    ],
                    dim=1,
                )
                tmp_value_states = torch.cat(
                    [
                        value_states[:, :num_attn_sinks],
                        value_states[:, num_attn_sinks + 1 :],
                    ],
                    dim=1,
                )
                attn_output = _flash_attention_forward(
                    query_states,
                    tmp_key_states,
                    tmp_value_states,
                    None,
                    q_len,
                    position_ids=position_ids,
                    dropout=dropout_rate,
                    sliding_window=None,
                    is_causal=self.is_causal,
                )
            else:
                # Prefilling with FlexAttention
                attn_output = self.flex_attn_forward(
                    bsz=bsz,
                    q_len=q_len,
                    attention_mask=attention_mask,
                    query_states=query_states,
                    key_states=key_states,
                    value_states=value_states,
                )
        else:
            # Full attention
            attn_output = _flash_attention_forward(
                query_states,
                key_states,
                value_states,
                None,  # set 'attention_mask' to None
                q_len,
                position_ids=position_ids,
                dropout=dropout_rate,
                sliding_window=None, # Hardcoded here
                is_causal=self.is_causal,
            )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()

        attn_weights = None
        if output_attentions:
            attn_weights = None

        if enable_ahn:
            return (
                attn_output,
                attn_weights,
                (query_states, key_states, value_states),
            )

        return attn_output, attn_weights


class Qwen3Model(Qwen3Model_):
    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        layer_cls = DECODER_LAYER_CLS[config._layer_implementation]
        self.layer_cls = config._layer_implementation
        self.layers = nn.ModuleList(
            [
                layer_cls(config, layer_idx, config._ahn_implementation)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)

        # Initialize weights and apply final processing
        self.post_init()

    @staticmethod
    def _is_fa_layer(
        config: Qwen3Config,
        decoder_layer: Optional[Qwen3DecoderLayer]
    ) -> Optional[bool]:
        fa_layer_ids = getattr(config, "fa_layer_ids", "")
        try:
            fa_layer_ids = [
                int(lid.strip())
                for lid in fa_layer_ids.split(",")
            ]
            enable_ahn = decoder_layer.layer_idx not in fa_layer_ids
            return enable_ahn
        except ValueError:
            return True
        except Exception as e:
            print(f"Unexpected error: {e}. Falling back to full attention for integrity.")
            return False

    @staticmethod
    def create_sparse_mask(sliding_window: int = 2048, num_attn_sinks: int = 0):

        def ctx_sliding_window_mask(b, h, q_idx, kv_idx):
            causal = q_idx >= kv_idx
            sliding = (q_idx - kv_idx) < sliding_window
            sink = kv_idx < num_attn_sinks
            return causal & (sliding | sink)

        return ctx_sliding_window_mask

    def pre_model_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None
    ):
        """
        Build FlexAttention mask with custom sliding window and attention sink.
        """
        flex_attention_mask = None
        if self.training and self.layer_cls == "Qwen3MemDecoderLayer":
            B, L = input_ids.shape
            sliding_window_type = getattr(self.config, "sliding_window_type", "fixed")
            sliding_window = getattr(self.config, "sliding_window", None)
            assert (
                sliding_window > 0
            ), "Please provide 'sliding_window' for 'Qwen3MemDecoderLayer' forward pass."
            
            # Get sliding window
            if sliding_window_type == "fixed":
                self.config.dy_sliding_window = sliding_window
            elif sliding_window_type == "linear":
                # Grow with sequence length but lower-bounded
                self.config.dy_sliding_window = max(L // 4, 512)
            elif sliding_window_type == "random":
                def sample_window_size(
                    seq_len: str,
                    candidates: list = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
                ):
                    valid_sizes = [c for c in candidates if seq_len / 8 < c < seq_len]
                    return random.choice(valid_sizes)
                random_size = sample_window_size(L)
                self.config.dy_sliding_window = random_size
            else:
                raise NotImplementedError
            
            # Get attention sink
            sliding_window = getattr(
                self.config, "dy_sliding_window", self.config.sliding_window
            )
            ahn_position = getattr(self.config, "ahn_position", "prefix")

            if ahn_position == "prefix":
                num_attn_sinks = 0
            elif ahn_position == "random":
                def sample_num_attn_sinks(
                    sliding_window: int,
                    candidates: list = [0, 32, 64, 128, 512, 2048, 4096],
                ):
                    valid_sizes = [c for c in candidates if c <= 0.5 * sliding_window]
                    return random.choice(valid_sizes)
                num_attn_sinks = sample_num_attn_sinks(sliding_window)
                self.config.dy_num_attn_sinks = num_attn_sinks
                sliding_window = sliding_window - num_attn_sinks
            else:
                raise NotImplementedError
        else:
            sliding_window = getattr(self.config, "sliding_window", None)
            num_attn_sinks = getattr(self.config, "num_attn_sinks", 0)
            assert (
                sliding_window > 0
            ), "Please provide 'sliding_window' for 'Qwen3MemDecoderLayer' forward pass."

        # Generate FlexAttention mask
        sparse_mask = self.create_sparse_mask(
            sliding_window=sliding_window, num_attn_sinks=num_attn_sinks
        )
        
        seq_len = input_ids.shape[1]
        padded_lens = [2048, 4096, 8192, 16384, 24576, 32768]
        padded_seq_len = next(
            (l for l in padded_lens if l >= seq_len), (seq_len // 128 + 1) * 128
        )

        flex_attention_mask = create_block_mask(
            sparse_mask,
            B=1,
            H=self.config.num_attention_heads,
            Q_LEN=padded_seq_len,
            KV_LEN=padded_seq_len,
            device=input_ids.device,
            BLOCK_SIZE=128,
            _compile=True,
        )

        return flex_attention_mask

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
        mem_past_key_values: Optional[Cache] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> CustomizedModelOutputWithPast:
        flex_attention_mask = self.pre_model_forward(input_ids=input_ids)

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            # logger.warning_once(
            #     "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            # )
            use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        # Note: Memory KV Cache
        if use_cache and not isinstance(mem_past_key_values, MemCache):
            if mem_past_key_values is None:
                mem_past_key_values = MemCache()
                # Empty hidden states cache
                for layer_id in range(len(self.layers)):
                    self.layers[layer_id].ahn.reset_cache()
            else:
                mem_past_key_values = MemCache.from_legacy_cache(mem_past_key_values)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # Note: Annotate for simplicity; use this only if the feature is required
        # causal_mask = self._update_causal_mask(
        #     attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        # )

        # Note: MemDecoderlayer is implemented with FlexAttention
        if attention_mask is not None and 0.0 in attention_mask:
            causal_mask = attention_mask
        else:
            causal_mask = None

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        
        # extra loss for layerwise distillation
        all_distill_pairs = []

        if self.training:
            mem_hidden_states = hidden_states.detach().clone()
            mem_hidden_states.requires_grad = True
        else:
            mem_hidden_states = None
        
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            # Note: Memory module on/off
            enable_ahn = self._is_fa_layer(self.config, decoder_layer)

            if self.gradient_checkpointing and self.training:
                decoder_layer.enable_ahn = False
                with torch.no_grad():
                    layer_outputs = decoder_layer(
                        hidden_states=hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        mem_past_key_values=mem_past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                    )
                decoder_layer.enable_ahn = enable_ahn
                if enable_ahn: # forward with memory module
                    mem_layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        mem_hidden_states,
                        flex_attention_mask,
                        position_ids,
                        past_key_values,
                        mem_past_key_values,
                        output_attentions,
                        use_cache,
                        cache_position,
                        position_embeddings,
                    )
            else:
                decoder_layer.enable_ahn = enable_ahn
                if enable_ahn:
                    layer_outputs = decoder_layer(
                        hidden_states=hidden_states,
                        attention_mask=flex_attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        mem_past_key_value=mem_past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states=hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        mem_past_key_values=mem_past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                    )
                
            if self.training:
                if "distill" in self.config.loss_type:
                    sliding_window = getattr(self.config, "dy_sliding_window", 0)
                    all_distill_pairs.append((
                        layer_outputs[1][:, sliding_window:, :],
                        mem_layer_outputs[1][:, sliding_window:, :])
                    )
                hidden_states, mem_hidden_states = layer_outputs[0], mem_layer_outputs[0]
            else:
                hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[2],)

        hidden_states = self.norm(hidden_states)

        if mem_hidden_states is not None:
            mem_hidden_states = self.norm(mem_hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = past_key_values if use_cache else None
        next_mem_cache = mem_past_key_values if use_cache else None

        return CustomizedModelOutputWithPast(
            last_hidden_states=hidden_states,
            last_mem_hidden_states=mem_hidden_states,
            past_key_values=next_cache,
            mem_past_key_values=next_mem_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            all_distill_pairs=all_distill_pairs
        )


class Qwen3ForCausalLM(Qwen3ForCausalLM_):
    def __init__(self, config, do_train: bool = False):
        super().__init__(config)
        self.model = Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def _filtered_state_dict(self, state_dict, pattern="ahn"):
        out = {}
        for name, param in state_dict.items():
            if pattern in name:
                out[name] = param
        return out
    
    def save_pretrained(self, save_directory, state_dict, **kwargs):
        if getattr(self.config, "save_ahn_only", False):
            state_dict = self._filtered_state_dict(state_dict)
            if not state_dict:
                raise ValueError("No AHN keys found.")
       
        super().save_pretrained(
            save_directory,
            state_dict=state_dict,
            safe_serialization=kwargs.pop("safe_serialization", True),
            max_shard_size=kwargs.pop("max_shard_size", "4GB"),
            **kwargs,
        )

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        do_train = kwargs.pop("do_train", False)
        model = super().from_pretrained(*args, **kwargs)

        def _init_weights(model: nn.Module):
            if (
                getattr(model.model, "layer_cls", "Qwen3DecoderLayer")
                == "Qwen3MemDecoderLayer"
            ):
                import deepspeed

                for name, param in model.named_parameters():
                    if "ahn.fn.q_proj" in name:
                        with deepspeed.zero.GatheredParameters(param, modifier_rank=0):
                            nn.init.normal_(param.data, mean=0.0, std=0.02)
                    if "ahn.fn.o_proj" in name:
                        with deepspeed.zero.GatheredParameters(param, modifier_rank=0):
                            nn.init.zeros_(param.data)
                    if "ahn_router.alpha" in name or "ahn_router.beta" in name:
                        with deepspeed.zero.GatheredParameters(param, modifier_rank=0):
                            nn.init.zeros_(param.data)
            else:
                model._init_weights()

        if do_train:
            _init_weights(model)
        return model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        mem_past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **loss_kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen3ForCausalLM

        >>> model = Qwen3ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            mem_past_key_values=mem_past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
        )

        hidden_states = outputs.last_hidden_states
        mem_hidden_states = outputs.last_mem_hidden_states

        logits_ref = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])
        logits_mem = (
            self.lm_head(mem_hidden_states[:, -num_logits_to_keep:, :])
            if mem_hidden_states is not None
            else mem_hidden_states
        )

        loss = ce_loss = kl_loss = distill_loss = None

        # Focus on tokens beyond the sliding window
        if self.training:
            sliding_window = getattr(
                self.config, "dy_sliding_window", self.config.sliding_window
            )
            valid_mask = torch.zeros_like(labels, dtype=bool)
            valid_mask[:, sliding_window:] = True
            valid_mask = valid_mask & (labels != -100)

            if (
                "kl" in self.config.loss_type
                and self.config._layer_implementation == "Qwen3MemDecoderLayer"
            ):
                ref_probs = F.softmax(logits_ref[valid_mask], dim=-1)
                mem_log_probs = F.log_softmax(logits_mem[valid_mask], dim=-1)

                kl_loss = F.kl_div(
                    mem_log_probs, ref_probs, reduction="none", log_target=False
                )
                kl_loss = kl_loss.sum() / valid_mask.sum()
                loss = loss + kl_loss if loss is not None else kl_loss

            if (
                "ce" in self.config.loss_type
                and self.config._layer_implementation == "Qwen3MemDecoderLayer"
            ):
                if labels is not None:
                    ce_loss = self.loss_function(
                        logits_mem[valid_mask], labels[valid_mask], self.vocab_size, **loss_kwargs
                    )
                    loss = loss + ce_loss if loss is not None else ce_loss

            if "distill" in self.config.loss_type:
                all_distill_pairs = outputs.all_distill_pairs
                targets, preds = map(torch.cat, zip(*all_distill_pairs)) # targets: (num_layers, seq_len, hidden_size)
                stacked_distill_loss = F.mse_loss(preds, targets, reduction="none").mean(dim=(-1, -2))
                distill_loss = stacked_distill_loss.mean()
                loss = loss + distill_loss if loss is not None else distill_loss

            world_size = dist.get_world_size() if dist.is_initialized() else None
            with torch.no_grad():
                if labels is not None:
                    # Reference CE loss (full attention)
                    ref_ce_loss = self.loss_function(
                        logits_ref[valid_mask],
                        labels[valid_mask],
                        self.vocab_size,
                        **loss_kwargs,
                    )
                    gathered_ref_ce = [
                        torch.zeros_like(ref_ce_loss.detach())
                        for _ in range(world_size)
                    ]
                    dist.all_gather(gathered_ref_ce, ref_ce_loss.detach())
                    if dist.get_rank() == 0:
                        ref_ce_loss = torch.stack(gathered_ref_ce).mean()
                        wandb.log({"reference ce loss": ref_ce_loss.item()})

                    # Memory CE loss (hybrid attention)
                    if ce_loss is None:
                        mem_ce_loss = self.loss_function(
                            logits_mem[valid_mask],
                            labels[valid_mask],
                            self.vocab_size,
                            **loss_kwargs,
                        )
                        gathered_mem_ce = [
                            torch.zeros_like(mem_ce_loss.detach())
                            for _ in range(world_size)
                        ]
                        dist.all_gather(gathered_mem_ce, mem_ce_loss.detach())

                        if dist.get_rank() == 0:
                            mem_ce_loss = torch.stack(gathered_mem_ce).mean()
                            wandb.log({"ce loss": mem_ce_loss.item()})

        if ce_loss is not None:
            gathered_ce = [
                torch.zeros_like(ce_loss.detach()) for _ in range(world_size)
            ]
            dist.all_gather(gathered_ce, ce_loss.detach())
            if dist.get_rank() == 0:
                mean_loss = torch.stack(gathered_ce).mean()
                wandb.log({"ce loss": mean_loss.item()})

        if kl_loss is not None:
            gathered_kl = [
                torch.zeros_like(kl_loss.detach()) for _ in range(world_size)
            ]
            dist.all_gather(gathered_kl, kl_loss.detach())
            if dist.get_rank() == 0:
                mean_loss = torch.stack(gathered_kl).mean()
                wandb.log({"kl loss": mean_loss.item()})

        if distill_loss is not None:
            gathered_distill = [
                torch.zeros_like(stacked_distill_loss.detach())
                for _ in range(world_size)
            ]
            dist.all_gather(gathered_distill, stacked_distill_loss.detach())
            if dist.get_rank() == 0:
                cum_mean_loss = torch.stack(gathered_distill).mean(dim=0)
                wandb.log({"average cum reg loss": cum_mean_loss.mean().item()})
                for i, cum_mean_loss_i in enumerate(cum_mean_loss):
                    wandb.log({f"layer {i:02d} cum reg loss": cum_mean_loss_i.item()})

        return CustomizedCausalLMOutputWithPast(
            loss=loss,
            logits=logits_ref,
            past_key_values=outputs.past_key_values,
            mem_past_key_values=outputs.mem_past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    # Adapted from 'transformers/generation/utils.py' for memory-llm forward
    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        # update past_key_values keeping its naming used in model code
        model_kwargs["past_key_values"] = getattr(outputs, "past_key_values")
        model_kwargs["mem_past_key_values"] = getattr(
            outputs, "mem_past_key_values", None
        )

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat(
                [token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1
            )

        if not is_encoder_decoder:
            # update attention mask
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [
                        attention_mask,
                        attention_mask.new_ones((attention_mask.shape[0], 1)),
                    ],
                    dim=-1,
                )
        else:
            # update decoder attention mask
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                model_kwargs["decoder_attention_mask"] = torch.cat(
                    [
                        decoder_attention_mask,
                        decoder_attention_mask.new_ones(
                            (decoder_attention_mask.shape[0], 1)
                        ),
                    ],
                    dim=-1,
                )

        if model_kwargs.get("use_cache", True):
            model_kwargs["cache_position"] = (
                model_kwargs["cache_position"][-1:] + num_new_tokens
            )
        else:
            past_positions = model_kwargs.pop("cache_position")
            new_positions = torch.arange(
                past_positions[-1] + 1,
                past_positions[-1] + num_new_tokens + 1,
                dtype=past_positions.dtype,
            ).to(past_positions.device)
            model_kwargs["cache_position"] = torch.cat((past_positions, new_positions))
        return model_kwargs


class Qwen3MemDecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
        ahn_cls_name: str = "GatedDeltaNet"
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        ahn_cls = AHN_CLS[ahn_cls_name]
        self.self_attn = Qwen3MemFlexAttn(config, layer_idx)

        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

        self.ahn = BaseAHN(
            ahn_cls=ahn_cls,
            layer_idx=layer_idx,
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            use_q_proj=getattr(config, "use_q_proj", False),
        )

        self.mlp = Qwen3MLP(config)
        
        if config.use_ahn_router:
            self.ahn_router = AHNRouter(
                num_heads=config.num_attention_heads,
                head_dim=head_dim,
            )
        else:
            self.ahn_router = None
        
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def trim_cache(
        self,
        past_key_values: List[torch.Tensor],
        layer_idx: int,
        num_attn_sinks: int,
        num_cached_tokens: int,
    ):
        """
        Evict key-value tokens from the cache after compression into memory.
        """
        assert (
            num_cached_tokens > 0
        ), "Cache should be trimmed only if num_cached_tokens > 0."
        past_key, past_value = past_key_values[layer_idx]
        past_key_values.key_cache[layer_idx] = torch.cat(
            [
                past_key[..., :num_attn_sinks, :],
                past_key[..., num_attn_sinks + num_cached_tokens :, :],
            ],
            dim=-2,
        ).contiguous()
        past_key_values.value_cache[layer_idx] = torch.cat(
            [
                past_value[..., :num_attn_sinks, :],
                past_value[..., num_attn_sinks + num_cached_tokens :, :],
            ],
            dim=-2,
        ).contiguous()  # B, H_kv, L, D
        
        # Trim memory cache
        self.ahn.trim_cache(
            num_attn_sinks=num_attn_sinks, num_cached_tokens=num_cached_tokens
        )

        return past_key_values

    @staticmethod
    def apply_rotary_emb_single(
        self,
        x: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        unsqueeze_dim: int = 1,
    ) -> torch.Tensor:
        cos, sin = position_embeddings
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        x_embed = (x * cos) + (rotate_half(x) * sin)
        return x_embed

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        mem_past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        # Note: Gradient checkpointing requirement.
        enable_ahn = getattr(self, "enable_ahn", False)

        if self.training:
            fn = self.mem_forward_train if enable_ahn else self.default_forward
        else:
            fn = self.mem_forward_inference if enable_ahn else self.default_forward

        return fn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            mem_past_key_value=mem_past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

    def default_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, attn_output = self.self_attn(
            enable_ahn=self.enable_ahn,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        if self.config.use_normalized_l2:
            attn_output = F.normalize(
                attn_output, dim=-1, p=2
            )
        
        outputs = (hidden_states, attn_output)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs

    # TODO: Complete annotations
    def mem_forward_train(
        self,
        hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # will become mandatory in v4.46
        dummy_size: Optional[int] = 128,
        **kwargs
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states
        prenormed_hidden_states = self.input_layernorm(hidden_states)

        (
            hidden_states, # not used
            self_attn_weights,
            attn_output,
            (q_states, k_states, v_states),
        ) = self.self_attn(
            enable_ahn=self.enable_ahn,
            hidden_states=prenormed_hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

        sliding_window = getattr(
            self.config, "dy_sliding_window", self.config.sliding_window
        )
        num_attn_sinks = getattr(self.config, "dy_num_attn_sinks", 0)

        assert sliding_window > 0, "'sliding_window' is required for AHN."
        seq_len = q_states.shape[1]  # q_states are in the shape of (B, L, H, D)
        in_ahn_seq_len = seq_len - sliding_window

        if in_ahn_seq_len > 0:
            # Linear kernel for q: project first, then apply ROPE
            if getattr(self.config, "use_q_proj", False):
                raw_q_states = self.self_attn.q_proj(prenormed_hidden_states)
                bsz, q_len = q_states.shape[:2]
                nope_q_states = self.ahn.fn.q_proj(raw_q_states)
                nope_q_states = nope_q_states.view(bsz, q_len, self.self_attn.num_heads, self.self_attn.head_dim).transpose(1, 2)
                q_states = self.apply_rotary_emb_single(
                    nope_q_states,
                    position_embeddings,
                )
                q_states = q_states.transpose(1, 2)
                
            # Cached tokens as input
            mem_h_ab = prenormed_hidden_states[
                :, num_attn_sinks : num_attn_sinks + in_ahn_seq_len, ...
            ].contiguous()
            mem_k = k_states[
                :, num_attn_sinks : num_attn_sinks + in_ahn_seq_len, ...
            ].contiguous()
            mem_v = v_states[
                :, num_attn_sinks : num_attn_sinks + in_ahn_seq_len, ...
            ].contiguous()

            # Current tokens as input
            mem_h_g = prenormed_hidden_states[
                :, sliding_window:, ...
            ].contiguous()
            mem_q = q_states[:, sliding_window:, ...].contiguous()

            ahn_attn_output, _ = self.ahn(
                hidden_states=[mem_h_ab, mem_h_g],
                q_states=mem_q,
                k_states=mem_k,
                v_states=mem_v,
            )

        # Avoid in-place operations
        ensemble_attn_output = attn_output.clone()
        if isinstance(self.ahn_router, AHNRouter):
            pos_ratio = (torch.arange(in_ahn_seq_len) + 1).to(
                mem_h_g.device
            ) / sliding_window
            ensemble_attn_output[:, sliding_window:, :] = self.ahn_router(
                pos_ratio=pos_ratio,
                h_mem=ahn_attn_output,
                h_local=attn_output[:, sliding_window:, :],
            )
        elif (
            isinstance(self.ahn_router, str)
            and self.ahn_router == "attention_score"
        ):
            raise NotImplementedError  # Impractical during inference
        else:
            ensemble_attn_output[:, sliding_window:, :] = (
                attn_output[:, sliding_window:, :] + ahn_attn_output
            )
        
        if self.config.use_normalized_l2:
            output_attn_output = F.normalize(
                ensemble_attn_output, dim=-1, p=2
            )
        else:
            output_attn_output = ensemble_attn_output

        hidden_states = self.self_attn.o_proj(ensemble_attn_output.contiguous())
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, output_attn_output)

        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs

    def mem_forward_inference(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        mem_past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = True,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states
        prenormed_hidden_states = self.input_layernorm(hidden_states)

        enable_ahn = getattr(self, "enable_ahn", False)
        # Attention forward without self.o_proj
        (
            attn_output, # Hardcoded, attn_output before attn::o_proj
            _,
            (q_states, k_states, v_states),
        ) = self.self_attn(
            enable_ahn=enable_ahn,
            hidden_states=prenormed_hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

        if enable_ahn:
            # [Warning] Hardcoded to enable sliding window
            sliding_window = getattr(self.config, "sliding_window", None)
            assert (
                sliding_window > 0
            ), "Please pass argument 'sliding_window' for class 'BaseAHN'."

            num_attn_sinks = getattr(self.config, "num_attn_sinks", 0)
            # KV cache is already updated, q_states: (B, L, H, D)
            cur_cache_size = (
                0
                if len(past_key_value) <= self.layer_idx
                else past_key_value[self.layer_idx][0].shape[2]
            )
            in_ahn_seq_len = cur_cache_size - sliding_window - num_attn_sinks

            self.ahn.update_cache(prenormed_hidden_states)

            if in_ahn_seq_len > 0:
                cached_gate_values = self.ahn.query_cache(
                    num_attn_sinks=num_attn_sinks, num_cached_toekns=in_ahn_seq_len
                )
                mem_h = [
                    cached_gate_values,
                    prenormed_hidden_states[:, -in_ahn_seq_len:, ...],
                ]
                
                if getattr(self.config, "use_q_proj", False):
                    bsz, q_len = q_states.shape[:2]
                    raw_q_states = self.self_attn.q_proj(prenormed_hidden_states)
                    nope_q_states = self.ahn.fn.q_proj(raw_q_states)
                    nope_q_states = nope_q_states.view(bsz, q_len, self.self_attn.num_heads, self.self_attn.head_dim).transpose(1, 2)
                    q_states = self.apply_rotary_emb_single(
                        nope_q_states,
                        position_embeddings,
                    )
                    q_states = q_states.transpose(1, 2)

                mem_q = q_states[:, -in_ahn_seq_len:, ...].contiguous()
                
                mem_k = k_states[
                    :, num_attn_sinks : num_attn_sinks + in_ahn_seq_len, ...
                ]
                mem_v = v_states[
                    :, num_attn_sinks : num_attn_sinks + in_ahn_seq_len, ...
                ]

                ahn_attn_output, mem_past_key_value = self.ahn(
                    hidden_states=mem_h,
                    q_states=mem_q,
                    k_states=mem_k,
                    v_states=mem_v,
                    use_cache=use_cache,
                    past_key_values=mem_past_key_value,
                )

                if isinstance(self.ahn_router, AHNRouter):
                    pos_ratio = (
                        self.ahn.num_cached_tokens
                        - torch.arange(in_ahn_seq_len - 1, -1, -1)
                    ).to(hidden_states.device) / sliding_window
                    hidden_states[:, -in_ahn_seq_len:, :] = self.ahn_router(
                        pos_ratio=pos_ratio,
                        h_mem=ahn_attn_output,
                        h_local=attn_output[:, -in_ahn_seq_len:, :],
                    )
                else:
                    attn_output[:, -in_ahn_seq_len:, :] = (
                        attn_output[:, -in_ahn_seq_len:, :] + ahn_attn_output
                    )

        hidden_states = self.self_attn.o_proj(attn_output)
        # Residual connection
        hidden_states = residual + hidden_states  # (B, L, D)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        # Notice: 'output_attentions' in Qwen3Model must be False during inference
        # if output_attentions:
        #     outputs += (self_attn_weights,)
        
        if enable_ahn and in_ahn_seq_len > 0 and past_key_value is not None:
            past_key_value = self.trim_cache(
                past_key_value,
                layer_idx=self.layer_idx,
                num_attn_sinks=num_attn_sinks,
                num_cached_tokens=in_ahn_seq_len,
            )

        return outputs


DECODER_LAYER_CLS = {
    "Qwen3DecoderLayer": Qwen3DecoderLayer,
    "Qwen3MemDecoderLayer": Qwen3MemDecoderLayer,
}


AHN_CLS = {
    "GatedDeltaNet": GatedDeltaNet,
    "DeltaNet": DeltaNet,
    "Mamba2": Mamba2,
}


def register_customized_qwen3(exist_ok=True):
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

    AutoConfig.register("qwen3", Qwen3Config, exist_ok=exist_ok)
    AutoModel.register(Qwen3Config, Qwen3Model, exist_ok=exist_ok)
    AutoModelForCausalLM.register(Qwen3Config, Qwen3ForCausalLM, exist_ok=exist_ok)