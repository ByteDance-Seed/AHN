# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

# This file has been modified by ByteDance Ltd. and/or its affiliates. on 2025-05-20.
#
# Original file was released under MIT, with the full license text
# available at https://github.com/fla-org/flash-linear-attention/blob/main/LICENSE.
#
# This modified file is released under Apache-2.0.

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from ..utils import GroupLinear
from fla.modules import FusedRMSNormSwishGate, RMSNorm, ShortConvolution
from fla.ops.common.utils import prepare_position_ids, prepare_sequence_ids
from fla.ops.delta_rule import chunk_delta_rule, fused_recurrent_delta_rule

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache


def elu_p1(x):
    return (F.elu(x, 1., False) + 1.).to(x)


def sum_norm(x):
    return (x / x.sum(-1, keepdim=True)).to(x)


class DeltaNet(nn.Module):
    r"""
    The layer implementaion for [Parallelizing Linear Transformers with the Delta Rule over Sequence Length](https://arxiv.org/abs/2406.06484).  # noqa:
    DeltaNet was originally proposed in [Linear Transformers Are Secretly Fast Weight Programmers](https://arxiv.org/abs/2102.11174). # noqa

    Args:
        mode (str, Optional):
            Which DeltaNet kernel to use.
            Currently available: `chunk`, `fused_recurrent`, and `fused_chunk`.
            Default: `chunk`.
        hidden_size (int, Optional):
            The hidden size of the input. Default: 1024.
        expand_k (float, Optional):
            The expansion ratio for the key dim. Default: 1.0.
        expand_v (float, Optional):
            The expansion ratio for the value dim. Default: 1.0.
        num_heads (int, Optional):
            The number of heads. Default: 4.
        use_beta (bool, Optional):
            Whether to use beta. Default: `True`.
        use_gate (bool, Optional):
            Whether to use output gate. Default: `False`.
        use_short_conv (bool, Optional):
            Whether to use short convolutions. Default: `True`.
        conv_size (int, Optional):
            The kernel size of the short convolution, only used when `use_short_conv` is `True`. Default: 4.
        conv_bias (bool, Optional):
            Whether to use bias in the short convolution, only used when `use_short_conv` is `True`. Default: `False`.
        allow_neg_eigval (bool, Optional):
            Allow negative eigenvalues. Default: `False`. If set to `True`, the beta will be multiplied by 2.
            See reference: [Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues](https://arxiv.org/abs/2411.12537)
        layer_idx (int, Optional):
            The index of the layer. Default: None.
        norm_eps (float, Optional):
            The epsilon value for the layernorm/rmsnorm layer. Default: 1e-5.
        qk_activation (str, Optional):
            The activation function for the query and key. Default: `silu`.
        qk_norm (str, Optional):
            The normalization method for the query and key. Default: `l2`.
    """

    def __init__(
        self,
        mode: str = 'chunk',
        d_model: int = None,
        hidden_size: int = 1024,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        num_heads: int = 4,
        head_dim: int = 128,
        use_beta: bool = True,
        use_gate: bool = True,
        use_short_conv: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        allow_neg_eigval: bool = False,
        layer_idx: int = None,
        qk_activation: str = 'silu',
        qk_norm: str = 'l2',
        norm_eps: float = 1e-5,
        use_shared_proj: bool = True,
        **kwargs
    ) -> DeltaNet:
        super().__init__()

        self.mode = mode
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm

        assert self.qk_activation in ['silu', 'relu', 'elu', 'identity']
        assert self.qk_norm in ['l2', 'sum']

        if d_model is not None:
            hidden_size = d_model
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.allow_neg_eigval = allow_neg_eigval

        self.key_dim = int(self.num_heads * self.head_dim * expand_k)
        self.value_dim = int(self.num_heads * self.head_dim * expand_v)
        self.head_k_dim = self.head_dim * expand_k
        self.head_v_dim = self.head_dim * expand_v
        self.layer_idx = layer_idx

        if mode == 'fused_chunk':
            raise NotImplementedError("fused_chunk_delta_rule is now deprecated. Please use `chunk_delta_rule` instead.")
        assert mode in ['chunk', 'fused_recurrent'], f"Not suppoerted mode `{mode}`."
        assert self.key_dim % num_heads == 0, f"key dim must be divisible by num_heads of {num_heads}"
        assert self.value_dim % num_heads == 0, f"value dim must be divisible by num_heads of {num_heads}"

        if not use_shared_proj:
            self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
            self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
            self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        self.use_beta = use_beta
        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size, self.num_heads, bias=False)
        if use_short_conv:
            self.conv_size = conv_size
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                activation='silu' if qk_activation == 'silu' else None
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                activation='silu' if qk_activation == 'silu' else None
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=conv_size,
                activation='silu'
            )
       
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.num_heads, bias=False)
            self.o_norm = FusedRMSNormSwishGate(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.hidden_size, eps=norm_eps)

        self.o_proj = GroupLinear(self.value_dim, self.num_heads * self.head_dim, self.num_heads)

    def get_beta_alpha(self, hidden_states: torch.Tensor):
        beta = self.b_proj(hidden_states).sigmoid()
        return beta, None
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        q_states: torch.Tensor,
        k_states: torch.Tensor,
        v_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs: Unpack[Dict]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens, position_ids, seq_idx = kwargs.get('cu_seqlens', None), kwargs.get('position_ids', None), None
      
        q = q_states
        k = k_states
        v = v_states
        
        hidden_states_ab, hidden_states_g = hidden_states

        if self.training:
            if self.use_beta:
                beta = self.b_proj(hidden_states_ab).sigmoid()
            else:
                beta = q.new_ones(q.shape[0], q.shape[1], q.shape[2])
        else:
            beta = hidden_states_ab["beta"]

        if self.allow_neg_eigval:
            beta = beta * 2.

        # dealing with padding
        if attention_mask is not None:
            beta = beta.mul(attention_mask[:, -beta.shape[-2]:, None])

        if self.training:
            mode = self.mode
            assert mode == 'chunk', "Only chunk mode is supported in training."
        else:
            q_len = q.shape[1]
            mode = 'fused_recurrent' if q_len <= 64 else self.mode

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None
        
        if mode == 'fused_recurrent':
            o, recurrent_state = fused_recurrent_delta_rule(
                q=q,
                k=k,
                v=v,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True if self.qk_norm == 'l2' else False,
                head_first=False,
            )
        elif mode == 'chunk':
            o, recurrent_state = chunk_delta_rule(
                q=q,
                k=k,
                v=v,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                head_first=False,
                use_qk_l2norm_in_kernel=True if self.qk_norm == 'l2' else False
            )

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=None,
                layer_idx=self.layer_idx,
                offset=q.shape[1]
            )

        if self.use_gate:
            g = rearrange(self.g_proj(hidden_states_g).repeat_interleave(self.head_v_dim, dim=-1), '... (h d) -> ... h d', d=self.head_v_dim)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj(o)

        return o, None, past_key_values
