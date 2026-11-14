import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple


class LoRALinear(nn.Module):
    """
    Parameter-Efficient Fine-Tuning LoRA layer for linear projections.
    W_out = W_base(x) + (alpha / r) * (x @ A.T @ B.T)
    """
    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 4,
        alpha: float = 8.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank if rank > 0 else 1.0

        # Freeze base layer weights
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        if rank > 0:
            dev = base_layer.weight.device
            dt = base_layer.weight.dtype
            # Down-projection A (initialized with Kaiming uniform)
            self.lora_A = nn.Parameter(torch.zeros(rank, self.in_features, device=dev, dtype=dt))
            # Up-projection B (initialized with zeros)
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank, device=dev, dtype=dt))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
            self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        else:
            self.lora_A = None
            self.lora_B = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        if self.rank > 0:
            lora_out = (self.dropout(x) @ self.lora_A.T) @ self.lora_B.T
            return base_out + self.scaling * lora_out
        return base_out


class LoRAConv1d(nn.Module):
    """
    Parameter-Efficient Fine-Tuning LoRA layer for 1D temporal convolutions.
    Uses 1x1 low-rank temporal convolutions.
    """
    def __init__(
        self,
        base_layer: nn.Conv1d,
        rank: int = 4,
        alpha: float = 8.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank if rank > 0 else 1.0

        # Freeze base layer
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        if rank > 0:
            dev = base_layer.weight.device
            dt = base_layer.weight.dtype
            self.lora_A = nn.Conv1d(base_layer.in_channels, rank, kernel_size=1, bias=False, device=dev, dtype=dt)
            self.lora_B = nn.Conv1d(rank, base_layer.out_channels, kernel_size=1, bias=False, device=dev, dtype=dt)
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)
            self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        else:
            self.lora_A = None
            self.lora_B = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        if self.rank > 0:
            lora_out = self.lora_B(self.lora_A(self.dropout(x)))
            return base_out + self.scaling * lora_out
        return base_out


class LoRAConv2d(nn.Module):
    """
    Parameter-Efficient Fine-Tuning LoRA layer for 2D convolutions.
    Uses 1x1 low-rank point convolutions.
    """
    def __init__(
        self,
        base_layer: nn.Conv2d,
        rank: int = 4,
        alpha: float = 8.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank if rank > 0 else 1.0

        # Freeze base layer
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        if rank > 0:
            dev = base_layer.weight.device
            dt = base_layer.weight.dtype
            self.lora_A = nn.Conv2d(base_layer.in_channels, rank, kernel_size=1, bias=False, device=dev, dtype=dt)
            self.lora_B = nn.Conv2d(rank, base_layer.out_channels, kernel_size=1, bias=False, device=dev, dtype=dt)
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)
            self.dropout = nn.Dropout2d(p=dropout) if dropout > 0.0 else nn.Identity()
        else:
            self.lora_A = None
            self.lora_B = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        if self.rank > 0:
            lora_out = self.lora_B(self.lora_A(self.dropout(x)))
            return base_out + self.scaling * lora_out
        return base_out


def apply_lora_to_model(
    model: nn.Module,
    rank: int = 4,
    alpha: float = 8.0,
    dropout: float = 0.0,
    target_names: Optional[List[str]] = None
) -> nn.Module:
    """
    Recursively replaces target linear and convolutional layers with LoRA layers.
    Freezes all non-LoRA parameters.
    """
    if target_names is None:
        target_names = ["to_q", "to_k", "to_v", "proj", "temporal_conv", "temporal_proj"]

    # First freeze all non-LoRA parameters
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False

    # Substitute target layers
    for name, module in list(model.named_children()):
        if any(t in name for t in target_names):
            if isinstance(module, (LoRALinear, LoRAConv1d, LoRAConv2d)):
                if module.lora_A is not None:
                    module.lora_A.weight.requires_grad = True
                if module.lora_B is not None:
                    module.lora_B.weight.requires_grad = True
            elif isinstance(module, nn.Linear):
                setattr(model, name, LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout))
            elif isinstance(module, nn.Conv1d):
                setattr(model, name, LoRAConv1d(module, rank=rank, alpha=alpha, dropout=dropout))
            elif isinstance(module, nn.Conv2d):
                setattr(model, name, LoRAConv2d(module, rank=rank, alpha=alpha, dropout=dropout))
        else:
            # Recurse
            apply_lora_to_model(module, rank=rank, alpha=alpha, dropout=dropout, target_names=target_names)

    return model


def get_trainable_parameters(model: nn.Module) -> Tuple[int, int, float]:
    """Returns (trainable_params, total_params, trainable_ratio_percent)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    ratio = (100.0 * trainable / total) if total > 0 else 0.0
    return trainable, total, ratio


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extracts only the trainable LoRA parameters from the model."""
    lora_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad and "lora_" in name:
            lora_dict[name] = param.data.clone()
    return lora_dict


def load_lora_state_dict(model: nn.Module, lora_dict: Dict[str, torch.Tensor]) -> None:
    """Loads LoRA parameters into the model."""
    model.load_state_dict(lora_dict, strict=False)
