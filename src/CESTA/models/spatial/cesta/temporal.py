from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class PatchGRUEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        if hidden_size < 16 or input_size < 1:
            raise ValueError("patch_gru requires hidden_size >= 16 and positive input_size")
        self.recurrent_size = (hidden_size * 7 // 8) // 8 * 8
        self.gru = nn.GRU(
            4 * input_size, self.recurrent_size, num_layers=2, batch_first=True, dropout=dropout, bidirectional=True
        )
        self.projection = nn.Linear(2 * self.recurrent_size, 2 * hidden_size)
        baseline_parameters = 6 * hidden_size * (input_size + 4 * hidden_size + 4)
        if sum(parameter.numel() for parameter in self.parameters()) > baseline_parameters:
            raise ValueError("patch_gru exceeds the equivalent two-layer GRU parameter budget")

    def patch_inputs(self, features: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if features.ndim != 3 or features.shape[1] < 1:
            raise ValueError("patch_gru requires nonempty (batch, time, features) inputs")
        if mask is not None and mask.shape != features.shape[:2]:
            raise ValueError("patch_gru mask must have shape (batch, time)")
        if mask is None:
            return F.pad(features, (0, 0, 3, 0)).unfold(1, 4, 1).transpose(-1, -2).flatten(2)
        valid = mask.bool()
        values = features.masked_fill(~valid.unsqueeze(-1), 0.0)
        patches = F.pad(values, (0, 0, 3, 0)).unfold(1, 4, 1).transpose(-1, -2)
        validity = F.pad(valid, (3, 0), value=False).unfold(1, 4, 1)
        contiguous = validity.flip(-1).to(torch.int32).cumprod(-1).flip(-1).bool()
        return patches.masked_fill(~contiguous.unsqueeze(-1), 0.0).flatten(2)

    def forward(self, features: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, None]:
        hidden, _ = self.gru(self.patch_inputs(features, mask))
        output = self.projection(hidden)
        if mask is not None:
            output = output.masked_fill(~mask.bool().unsqueeze(-1), 0.0)
        return output, None


class GRUDilatedConvEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        width = 2 * hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(width)
        self.branches = nn.ModuleList([
            nn.Conv1d(width, width, kernel_size=3, padding=dilation, dilation=dilation, groups=width)
            for dilation in (1, 4)
        ])
        self.projection = nn.Conv1d(width, width, kernel_size=1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, None]:
        hidden, _ = self.gru(features)
        inputs = self.norm(self.dropout(hidden)).transpose(1, 2)
        mixed = (self.branches[0](inputs) + self.branches[1](inputs)) * 0.5
        residual = self.projection(F.gelu(mixed)).transpose(1, 2)
        return hidden + self.dropout(residual), None


class PatchTSTTemporalEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        context_length: int = 60,
        d_model: int = 64,
        num_layers: int = 3,
        num_heads: int = 8,
        ffn_dim: int = 64,
        patch_length: int = 4,
        patch_stride: int = 1,
        dropout: float = 0.06,
        standalone_compatible: bool = False,
    ) -> None:
        super().__init__()
        from transformers import PatchTSTConfig
        from transformers.models.patchtst.modeling_patchtst import PatchTSTModel

        if not 1 <= patch_length <= context_length or patch_stride < 1:
            raise ValueError("Patch length must fit context_length and patch stride must be positive")
        if d_model < 1 or num_heads < 1 or d_model % num_heads or num_layers < 1 or ffn_dim < 1:
            raise ValueError("Invalid PatchTST dimensions or attention head count")
        self.standalone_compatible = standalone_compatible
        self.context_length = context_length
        self.encoder = PatchTSTModel(PatchTSTConfig.from_dict({
            "num_input_channels": input_size,
            "context_length": context_length,
            "patch_length": patch_length,
            "patch_stride": patch_stride,
            "d_model": d_model,
            "num_hidden_layers": num_layers,
            "num_attention_heads": num_heads,
            "ffn_dim": ffn_dim,
            "attention_dropout": dropout,
            "ff_dropout": dropout,
            "positional_dropout": dropout,
            "norm_type": "batchnorm" if standalone_compatible else "layernorm",
            "scaling": "std" if standalone_compatible else None,
            "do_mask_input": False,
        }))
        self.layer_norm = nn.LayerNorm(input_size * d_model)
        self.projection = nn.Linear(input_size * d_model, output_size)

    def train(self, mode: bool = True) -> PatchTSTTemporalEncoder:
        super().train(mode)
        if self.standalone_compatible:
            for module in self.modules():
                if isinstance(module, nn.BatchNorm1d):
                    module.eval()
        return self

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, None]:
        self.train(self.training)
        if features.shape[1] != self.context_length:
            raise ValueError("PatchTST input length must equal configured context_length")
        hidden = self.encoder(past_values=features).last_hidden_state
        hidden = hidden.permute(0, 2, 1, 3).flatten(2)
        hidden = F.interpolate(hidden.transpose(1, 2), size=features.shape[1], mode="linear", align_corners=False).transpose(1, 2)
        return self.projection(self.layer_norm(hidden)), None
