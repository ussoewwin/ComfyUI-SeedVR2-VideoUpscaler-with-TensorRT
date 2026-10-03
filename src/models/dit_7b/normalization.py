# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

from typing import Callable, Optional
from diffusers.models.normalization import RMSNorm
from torch import nn
import torch
import torch.nn.functional as F
import numbers
from torch.nn.parameter import Parameter
from torch.nn import init
import os as _os

# (dim: int, eps: float, elementwise_affine: bool)
norm_layer_type = Callable[[int, float, bool], nn.Module]


class CustomLayerNorm(nn.Module):
    """
    Custom LayerNorm implementation to replace Apex FusedLayerNorm
    """
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(CustomLayerNorm, self).__init__()
        
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if self.elementwise_affine:
            self.weight = Parameter(torch.Tensor(*normalized_shape))
            self.bias = Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input):
        return F.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps)


class CustomRMSNorm(nn.Module):
    """
    Custom RMSNorm implementation to replace Apex FusedRMSNorm
    """
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(CustomRMSNorm, self).__init__()
        
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if self.elementwise_affine:
            self.weight = Parameter(torch.ones(*normalized_shape))
        else:
            self.register_parameter('weight', None)

    def forward(self, input):
        # RMS normalization: x / sqrt(mean(x^2) + eps) * weight
        dims = tuple(range(-len(self.normalized_shape), 0))
        
        # _SEEDVR2_NORM_BF16_SWITCH: env SEEDVR2_NORM_BF16=1 -> bf16 path (VRAM-saving),
        # otherwise the stock fp32 path (quality-priority).
        if _os.environ.get("SEEDVR2_NORM_BF16") == "1":
            # bf16 path: avoid autocast fp32 promotion of pow/mean, and match weight dtype
            variance = torch.mean(input * input, dim=dims, keepdim=True, dtype=input.dtype)
            rms = torch.sqrt(variance + self.eps)
            normalized = input / rms
            if self.elementwise_affine:
                w = self.weight
                if w.dtype != input.dtype:
                    w = w.to(input.dtype)
                return normalized * w
            return normalized
        # stock fp32 path
        variance = input.pow(2).mean(dim=dims, keepdim=True)
        rms = torch.sqrt(variance + self.eps)
        normalized = input / rms
        if self.elementwise_affine:
            if hasattr(torch, "float8_e4m3fn"):
                fp8_types = (torch.float8_e4m3fn, torch.float8_e5m2)
                if self.weight.dtype in fp8_types:
                    return normalized * self.weight.to(input.dtype)
            return normalized * self.weight
        return normalized


def get_norm_layer(norm_type: Optional[str]) -> norm_layer_type:

    def _norm_layer(dim: int, eps: float, elementwise_affine: bool):
        if norm_type is None:
            return nn.Identity()

        if norm_type == "layer":
            return nn.LayerNorm(
                normalized_shape=dim,
                eps=eps,
                elementwise_affine=elementwise_affine,
            )

        if norm_type == "rms":
            return RMSNorm(
                dim=dim,
                eps=eps,
                elementwise_affine=elementwise_affine,
            )

        if norm_type == "fusedln":
            # Use custom LayerNorm instead of Apex FusedLayerNorm
            return CustomLayerNorm(
                normalized_shape=dim,
                elementwise_affine=elementwise_affine,
                eps=eps,
            )

        if norm_type == "fusedrms":
            # Use custom RMSNorm instead of Apex FusedRMSNorm
            return CustomRMSNorm(
                normalized_shape=dim,
                elementwise_affine=elementwise_affine,
                eps=eps,
            )

        raise NotImplementedError(f"{norm_type} is not supported")

    return _norm_layer
