# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn

from typing import Optional
from transformers.cache_utils import Cache

def register_custom_accelerator():
    from accelerate import Accelerator

    # Store the original method to call it later
    original_backward = Accelerator.backward

    # Define the new method
    def new_backward(self, loss, retain_graph=True, **kwargs):
        """
        Custom backward function with retain_graph=True by default.
        Calls the original backward method.
        """
        kwargs.setdefault("retain_graph", retain_graph)  # Ensure retain_graph is set
        original_backward(self, loss, **kwargs)  # Call the original method

    # Replace the original method
    Accelerator.backward = new_backward


class GroupLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_groups: int,
        use_bias: bool = False,
    ):
        super().__init__()
        assert (
            in_features % num_groups == 0
        ), "in_features must be divisible by num_groups"
        assert (
            out_features % num_groups == 0
        ), "out_features must be divisible by num_groups"
        self.use_bias = use_bias
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.group_in = in_features // num_groups
        self.group_out = out_features // num_groups

        self.weight = nn.Parameter(
            torch.Tensor(self.num_groups, self.group_in, self.group_out)
        )
        self.bias = (
            nn.Parameter(torch.zeros(self.num_groups, self.group_out))
            if self.use_bias
            else None
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.zero_()

    def forward(self, x):
        B, L, _ = x.shape  # (B, L, in_features)
        x = x.view(B, L, self.num_groups, self.group_in)  # (B, L, num_groups, group_in)
        x = x.reshape(
            -1, self.num_groups, self.group_in
        )  # (B * L, num_groups, group_in)
        out = torch.einsum(
            "bgi,gio->bgo", x, self.weight
        )  # (B * L, num_groups, group_out)
        if self.use_bias:
            out = out + self.bias
        out = out.reshape(
            B, L, self.num_groups * self.group_out
        )  # (B, L, out_features)
        return out


class BaseAHN(nn.Module):
    def __init__(
        self,
        ahn_cls: nn.Module,
        layer_idx: int,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        mode: str = "chunk",
        **kwargs
    ):
        """
        Naive constant-space memory following the recurrence formula M_t = f(M_{t-1}, q).
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.fn = ahn_cls(
            layer_idx=layer_idx,
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            mode=mode,
            expand_v=1,
            **kwargs
        )
        self.num_cached_tokens = 0  # Cached K/V token count
        self.ahn_cache = {}  # Keys: hidden_states, beta, alpha

    def update_cache(self, hidden_states: torch.Tensor):
        # Store the gate values only
        beta, alpha = self.fn.get_beta_alpha(hidden_states)
        if not self.ahn_cache:
            self.ahn_cache = {
                "beta": beta,
                "alpha": alpha,
            }
        else:
            self.ahn_cache = {
                "beta": torch.cat([self.ahn_cache["beta"], beta], dim=1),
                "alpha": torch.cat([self.ahn_cache["alpha"], alpha], dim=1) if alpha is not None else alpha,
            }

    def query_cache(self, num_attn_sinks: int, num_cached_toekns: int):
        memory_cache = self.ahn_cache
        gate_values = {
            "beta": memory_cache["beta"][
                :, num_attn_sinks : num_attn_sinks + num_cached_toekns, ...
            ],
            "alpha": memory_cache["alpha"][
                :, num_attn_sinks : num_attn_sinks + num_cached_toekns, ...
            ] if memory_cache["alpha"] is not None else None,
        }

        return gate_values

    def trim_cache(self, num_attn_sinks: int, num_cached_tokens: int):
        memory_cache = self.ahn_cache
        memory_cache["beta"] = torch.concat(
            [
                memory_cache["beta"][:, :num_attn_sinks, ...],
                memory_cache["beta"][:, num_attn_sinks + num_cached_tokens :, ...],
            ],
            dim=-2,
        ).contiguous()
        memory_cache["alpha"] = torch.concat(
            [
                memory_cache["alpha"][:, :num_attn_sinks, ...],
                memory_cache["alpha"][:, num_attn_sinks + num_cached_tokens :, ...],
            ],
            dim=-2,
        ).contiguous() if memory_cache["alpha"] is not None else None

    def reset_cache(self):
        self.num_cached_tokens = 0
        self.ahn_cache = {}

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_states: torch.Tensor,
        k_states: torch.Tensor,
        v_states: torch.Tensor,
        use_cache: bool = False,
        past_key_values: Optional[Cache] = None,
    ):
        o, _, past_key_values = self.fn(
            layer_idx=self.layer_idx,
            hidden_states=hidden_states,
            q_states=q_states,
            k_states=k_states,
            v_states=v_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
        )

        if not self.training:
            self.num_cached_tokens += o.shape[1]

        return o, past_key_values


class AHNRouter(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, use_dimwise_pos: bool = False):
        super().__init__()
        self.H, self.D = num_heads, head_dim

        self.alpha = nn.Parameter(torch.full((num_heads,), 0.2))  # slope
        self.beta = nn.Parameter(torch.zeros(num_heads))  # bias

        if use_dimwise_pos:
            self.freq = nn.Parameter(  # fixed frequencies initialised as in RoPE
                torch.randn(num_heads, head_dim) / math.sqrt(head_dim),
                requires_grad=False,
            )
            self.scale = nn.Parameter(torch.zeros(num_heads, head_dim))  # learnable amp

    def forward(
        self, pos_ratio: torch.FloatTensor, h_mem: torch.Tensor, h_local: torch.Tensor
    ):
        B, L = h_mem.shape[:2]
        H, D = self.H, self.D

        log_r = torch.log(pos_ratio).unsqueeze(0).unsqueeze(-1)  # B L 1
        pos = self.alpha * log_r + self.beta  # B L H

        if hasattr(self, "freq"):  # dim-wise enrichment
            sin_feature = torch.sin(log_r * self.freq)  # B L H D
            pos += (sin_feature * self.scale).sum(-1)  # B L H

        g = torch.sigmoid(pos).unsqueeze(-1)  # B L H 1
        out = h_local.reshape(B, L, H, D) + g * h_mem.reshape(B, L, H, D)  # B L H D
        return out.reshape(B, L, H * D).contiguous()