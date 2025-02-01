import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math

class DynamicQuantConfig:
    """
    Configuration for dynamic quantization with outlier-aware low-rank adaptation.
    
    Args:
        bits (int): Number of bits for activation quantization (e.g., 4).
        threshold (float): Outlier threshold; if score > threshold, apply LoRA-scale adaptation.
        adapt_rank (int): Rank of the low-rank adaptation (currently only rank=1 is shown).
        num_power_iters (int): Number of power-iteration steps to approximate max singular value.
        extra_scaling (float): Additional global scaling factor (if needed) for outlier channels.
    """
    def __init__(self,
                 bits: int = 4,
                 threshold: float = 0.35,
                 adapt_rank: int = 1,
                 num_power_iters: int = 2,
                 extra_scaling: float = 0.5):
        self.bits = bits
        self.threshold = threshold
        self.adapt_rank = adapt_rank
        self.num_power_iters = num_power_iters
        self.extra_scaling = extra_scaling


class DynamicQuantizer(nn.Module):
    """
    Performs dynamic outlier detection and optional low-rank scaling before quantization.
    Applies standard linear quantization for non-outlier activations.
    """
    def __init__(self, hidden_dim: int, config: DynamicQuantConfig):
        super().__init__()
        self.config = config
        self.bits = config.bits
        self.threshold = config.threshold
        self.num_power_iters = config.num_power_iters
        self.extra_scaling = config.extra_scaling

        # Example: rank-1 LoRA-style parameters.
        # For a hidden_dim 'd', we store 2 vectors of shape (d,)
        # which form a rank-1 update:  (X * lora_a) * lora_b
        # If adapt_rank > 1, you'd store multiple rank-1 factors.
        self.lora_a = nn.Parameter(torch.zeros(hidden_dim), requires_grad=True)
        self.lora_b = nn.Parameter(torch.zeros(hidden_dim), requires_grad=True)

        # Initialize them (small random values)
        nn.init.normal_(self.lora_a, std=0.02)
        nn.init.normal_(self.lora_b, std=0.02)

        # Precompute the quant range for the chosen bit-width
        # For 4-bit signed, range is -8..7
        # For 8-bit signed, range is -128..127, etc.
        self.register_buffer('quant_min', torch.tensor(-(2 ** (self.bits - 1))))
        self.register_buffer('quant_max', torch.tensor(2 ** (self.bits - 1) - 1))

    @torch.no_grad()
    def power_iteration(self, X: Tensor) -> float:
        """
        Approximates the largest singular value of X using power iteration.
        (Used for computing the outlier score)
        """
        device = X.device
        # Initialize v with the size of the feature dimension and same dtype as X
        v = torch.randn(X.shape[1], device=device, dtype=X.dtype)
        for _ in range(self.num_power_iters):
            u = X @ v  # X: (N, hidden_dim), v: (hidden_dim,)  => u: (N,)
            u = F.normalize(u, dim=0, eps=1e-8)
            v = X.transpose(0, 1) @ u  # v: (hidden_dim,)
            v = F.normalize(v, dim=0, eps=1e-8)
        sigma_max = (X @ v).norm(2)
        return sigma_max.item()

    @torch.no_grad()
    def compute_outlier_score(self, X: Tensor) -> float:
        """
        Computes the outlier score = sigma_max^2 / ||X||_F^2, where sigma_max is approximated
        via power iteration.
        """
        sigma_max = self.power_iteration(X)
        frob_norm_sq = torch.norm(X, p='fro').square().item()
        score = (sigma_max * sigma_max) / (frob_norm_sq + 1e-8)
        return score

    def forward(self, X: Tensor) -> Tensor:
        """
        1. Flattens the activation tensor to 2D (N, feature)
        2. Computes outlier score and applies rank-1 LoRA adaptation if needed.
        3. Applies quantization (and dequantizes back to floating-point).
        4. Reshapes the tensor back to its original shape.
        """
        orig_shape = X.shape
        # Flatten all dimensions except the last one (feature dimension)
        X_flat = X.view(-1, orig_shape[-1])

        # (A) Compute outlier score on flattened activations
        with torch.no_grad():
            score = self.compute_outlier_score(X_flat)

        # (B) If outlier, apply rank-1 adaptation
        if score > self.threshold:
            # LoRA update: delta = (X_flat * lora_a) @ diag(lora_b)
            # Here, X_flat: (N, hidden_dim), lora_a: (hidden_dim,)
            delta = X_flat * self.lora_a  # elementwise multiplication (broadcast over N)
            delta = delta @ torch.diag(self.lora_b)
            delta = delta * self.extra_scaling
            X_flat = X_flat + delta

        # (C) Quantization (per-tensor)
        max_val = X_flat.abs().max()
        if max_val < 1e-8:
            scale = 1.0
        else:
            scale = max_val / float(self.quant_max.item())

        X_int = torch.round(X_flat / scale)
        X_int = torch.clamp(X_int, self.quant_min, self.quant_max)
        X_quant = X_int * scale

        # Reshape back to the original shape
        return X_quant.view(orig_shape)
