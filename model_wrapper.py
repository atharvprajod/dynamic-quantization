#!/usr/bin/env python3
# model_wrappers.py

import torch
import torch.nn as nn
from torch import Tensor

from dynamic_quant import DynamicQuantizer, DynamicQuantConfig

class OutlierAwareLinear(nn.Module):
    """
    Example linear layer wrapper that:
    1) Applies an `nn.Linear` transform.
    2) Then passes the output activations through a `DynamicQuantizer`.
    """
    def __init__(self, in_features: int, out_features: int, config: DynamicQuantConfig):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dynamic_quant = DynamicQuantizer(hidden_dim=out_features, config=config)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        x = self.dynamic_quant(x)
        return x

def apply_dynamic_quant(model: nn.Module, config: DynamicQuantConfig):
    """
    Recursively walk the model. Whenever we find an nn.Linear,
    replace it with OutlierAwareLinear that has dynamic quantization.
    """
    for name, module in list(model.named_children()):
        if isinstance(module, nn.Linear):
            # Replace with our custom wrapper
            new_module = OutlierAwareLinear(module.in_features, module.out_features, config)
            # Copy existing linear weights/bias
            new_module.linear.weight.data = module.weight.data.clone()
            if module.bias is not None:
                new_module.linear.bias.data = module.bias.data.clone()
            setattr(model, name, new_module)
        else:
            # Recurse
            apply_dynamic_quant(module, config)

    return model
