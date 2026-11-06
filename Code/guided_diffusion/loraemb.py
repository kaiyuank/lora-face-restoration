# lora.py

import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    # LoRA for nn.Linear layers.
    def __init__(self, base_layer: nn.Linear, r: int = 4, alpha: float = 4.0, dropout: float = 0.0):
        super().__init__()
        assert isinstance(base_layer, nn.Linear), "LoRALinear requires an nn.Linear layer"
        self.base = base_layer
        self.r = r
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.scaling = 1.0

        in_dim, out_dim = base_layer.in_features, base_layer.out_features
        self.lora_A = nn.Parameter(torch.zeros(r, in_dim))
        self.lora_B = nn.Parameter(torch.zeros(out_dim, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return self.base(x) + self.scaling * (self.dropout(x) @ self.lora_A.T @ self.lora_B.T)

"""
class LoRAConv2d(nn.Module):
    # LoRA wrapper for nn.Conv2d layers.
    def __init__(self, base_layer: nn.Conv2d, r: int = 4, alpha: float = 1.0, dropout: float = 0.0):
        super().__init__()
        assert isinstance(base_layer, nn.Conv2d), "LoRAConv2d requires an nn.Conv2d layer"
        assert base_layer.kernel_size == (1, 1), "Only 1x1 conv supported for LoRA"
        self.base = base_layer
        self.r = r
        self.alpha = alpha
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.scaling = alpha / r

        in_dim = base_layer.in_channels
        out_dim = base_layer.out_channels
        self.lora_A = nn.Parameter(torch.zeros(r, in_dim, 1, 1))
        self.lora_B = nn.Parameter(torch.zeros(out_dim, r, 1, 1))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        base_out = self.base(x)
        lora_out = nn.functional.conv2d(self.dropout(x), self.lora_A)
        lora_out = nn.functional.conv2d(lora_out, self.lora_B)
        return base_out + self.scaling * lora_out
"""
'''
class LoRAConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=4, stride=1, padding=0, bias=True, scaling=0.25):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        for param in self.conv.parameters():
            param.requires_grad = False
        self.weight = self.conv.weight
        self.bias = self.conv.bias

        self.r = r
        self.scaling = scaling
        if r > 0:
            self.lora_down = nn.Conv1d(in_channels, r, kernel_size=1, bias=False)
            self.lora_up = nn.Conv1d(r, out_channels, kernel_size=1, bias=False)
            nn.init.zeros_(self.lora_up.weight)
            nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        else:
            self.lora_down = None
            self.lora_up = None
'''

class LoRAConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=4, stride=1, padding=0, bias=True, scaling=1.0):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        for param in self.conv.parameters():
            param.requires_grad = False
        
        self.weight = self.conv.weight
        self.bias = self.conv.bias
        
        self.r = r
        self.scaling = scaling
        if r > 0:
            self.lora_down = nn.Conv1d(in_channels, r, kernel_size=1, bias=False)
            self.lora_up = nn.Conv1d(r, out_channels, kernel_size=1, bias=False)
            nn.init.zeros_(self.lora_up.weight)
            nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        else:
            self.lora_down = None
            self.lora_up = None


    def forward(self, x):
        result = self.conv(x)
        if self.r > 0:
            lora_result = self.lora_up(self.lora_down(x))
            result = result + self.scaling*lora_result
        return result

def mark_only_lora_as_trainable(model: nn.Module):
    # Freezes all model parameters and just train LoRA layers.
    for name, param in model.named_parameters():
        param.requires_grad = 'lora_' in name or 'lora' in name


def get_lora_params(model: nn.Module):
    return (param for name, param in model.named_parameters() if 'lora_' in name or 'lora' in name)

def get_lora_state_dict(model: nn.Module):
    return {k: v for k, v in model.state_dict().items() if 'lora_' in k or 'lora' in k} 


