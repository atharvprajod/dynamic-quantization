import torch
import torch.nn as nn
from torch import Tensor

from dynamic_quant import DynamicQuantizer, DynamicQuantConfig

class OutlierAwareLinear(nn.Module):
    """
    A linear layer wrapper that:
      1) Applies an nn.Linear transform.
      2) Passes the output activations through a DynamicQuantizer.
    Both modules are moved to fp16 to avoid dtype mismatches.
    """
    def __init__(self, in_features: int, out_features: int, config: DynamicQuantConfig):
        super().__init__()
        # Create layers and immediately cast them to fp16
        self.linear = nn.Linear(in_features, out_features).to(torch.float16)
        self.dynamic_quant = DynamicQuantizer(hidden_dim=out_features, config=config).to(torch.float16)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        x = self.dynamic_quant(x)
        return x

def apply_dynamic_quant(model: nn.Module, config: DynamicQuantConfig):
    """
    Recursively traverse the model, and for every nn.Linear layer, replace it
    with an OutlierAwareLinear. This function also copies weights (and casts them
    to fp16) so that there is no dtype mismatch.
    """
    for name, module in list(model.named_children()):
        if isinstance(module, nn.Linear):
            # Create our custom module
            new_module = OutlierAwareLinear(module.in_features, module.out_features, config)
            # Copy and cast the linear weights and bias to fp16
            new_module.linear.weight.data = module.weight.data.clone().to(torch.float16)
            if module.bias is not None:
                new_module.linear.bias.data = module.bias.data.clone().to(torch.float16)
            setattr(model, name, new_module)
        else:
            # Recurse for nested modules
            apply_dynamic_quant(module, config)
    return model
