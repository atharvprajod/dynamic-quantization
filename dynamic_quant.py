#!/usr/bin/env python3
# dynamic_quant.py

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
        Computes an approximation of the largest singular value of X
        using power iteration. We do not need the actual singular vector
        for the outlier score, just sigma_max^2.

        For outlier score, we want sigma_max^2 / ||X||_F^2.
        """
        # X is shape (batch, hidden_dim) or (seq_len, hidden_dim).
        # We want the largest singular value of (X), or equivalently sqrt of largest eigenvalue of X^T X or X X^T.
        # We'll just do power iteration on X directly.
        device = X.device
        # v: random vector of shape (hidden_dim,)
        v = torch.randn(X.shape[1], device=device)
        for _ in range(self.num_power_iters):
            # u = X * v
            u = X @ v
            u = F.normalize(u, dim=0, eps=1e-8)

            # v = X^T * u
            v = X.transpose(0, 1) @ u
            v = F.normalize(v, dim=0, eps=1e-8)

        # Approx largest singular value = || X v ||_2
        sigma_max = (X @ v).norm(2)
        return sigma_max.item()

    @torch.no_grad()
    def compute_outlier_score(self, X: Tensor) -> float:
        """
        Outlier score = sigma_max^2 / ||X||_F^2
        where sigma_max is approximated via power iteration.
        """
        sigma_max = self.power_iteration(X)  # largest singular value
        frob_norm_sq = torch.norm(X, p='fro').square().item()
        # sigma_max^2
        sigma_max_sq = sigma_max * sigma_max
        score = sigma_max_sq / (frob_norm_sq + 1e-8)
        return score

    def forward(self, X: Tensor) -> Tensor:
        """
        1. Detect if X has outliers (via outlier score).
        2. If outlier_score > threshold, apply rank-1 LoRA scaling.
        3. Quantize X (adapted or not).
        4. Return (floating) dequantized X for subsequent layers (fake-quant approach).
        
        For real inference on integer hardware, you'd keep X in integer form.
        """
        # (A) Compute outlier score
        with torch.no_grad():
            score = self.compute_outlier_score(X)

        # (B) If outlier, apply rank-1 adaptation
        if score > self.threshold:
            # rank-1 update: delta = (X * lora_a) * lora_b
            # shape checks: X: (batch, d), lora_a: (d,), lora_b: (d,)
            # X * lora_a -> broadcast mul -> shape (batch, d)
            delta = X * self.lora_a  # broadcast over batch dimension
            # (delta) * lora_b means each row i in delta is scaled by lora_b
            # but we want an outer product effect. We'll do an elementwise along 'd':
            delta = delta @ torch.diag(self.lora_b)
            # Add an extra global scaling if desired
            delta = delta * self.extra_scaling
            X = X + delta

        # (C) Standard quantization (per-tensor for simplicity):
        # get scale and zero point
        # Example: scale = max_abs / (2^(bits-1) - 1)
        max_val = X.abs().max()
        if max_val < 1e-8:
            # Edge case: if everything is near-zero
            scale = 1.0
        else:
            scale = max_val / float(self.quant_max.item())

        # quantize
        X_int = torch.round(X / scale)
        X_int = torch.clamp(X_int, self.quant_min, self.quant_max)

        # "Dequantize" for further floating-point processing in PyTorch
        # (In a real system, you might keep it in int form until final output.)
        X_quant = X_int * scale  # plus zero_point if needed, but here we assume symmetrical

        return X_quant
