from __future__ import annotations

from copy import deepcopy
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F

from CESTA.models.spatial.stgcn.model import STConvBlock, STGCNClassifier, TemporalConv


class TemporalProjection(nn.Module):
    def __init__(self, layer: TemporalConv) -> None:
        super().__init__()
        conv = layer.conv
        norm = layer.bn
        if norm.running_var is None or norm.running_mean is None or conv.bias is None:
            raise ValueError("frozen BatchNorm statistics required")
        scale = norm.weight.detach() / torch.sqrt(norm.running_var + norm.eps)
        self.register_buffer("weight", conv.weight.detach().squeeze(2) * scale[:, None, None])
        self.register_buffer("bias", (conv.bias.detach() - norm.running_mean) * scale + norm.bias.detach())
        self.padding = cast(tuple[int, int], conv.padding)[1]

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        zeros = torch.zeros_like(values[:, : self.padding])
        padded = torch.cat([zeros, values, zeros], dim=1)
        weight = cast(torch.Tensor, self.weight)
        output = F.linear(padded[:, : values.shape[1]], weight[..., 0], cast(torch.Tensor, self.bias))
        for shift in range(1, weight.shape[-1]):
            output = output + F.linear(padded[:, shift : shift + values.shape[1]], weight[..., shift])
        return output


class STGCNPrepareStage(nn.Module):
    def __init__(self, model: STGCNClassifier, block_index: int) -> None:
        super().__init__()
        block = cast(STConvBlock, model.blocks[block_index])
        self.temporal = TemporalProjection(block.temporal1)
        self.input_width = model.features_per_node if block_index == 0 else model.st_hidden_size
        self.output_width = model.st_hidden_size

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.temporal(values).relu()


class STGCNGraphStage(nn.Module):
    def __init__(self, model: STGCNClassifier, receiver: int, block_index: int) -> None:
        super().__init__()
        block = cast(STConvBlock, model.blocks[block_index])
        edges = cast(torch.Tensor, model.edge_index)
        selected = (edges[1] == receiver) & (edges[0] != receiver)
        self.sender_indices = edges[0, selected].tolist()
        self.edge_indices = selected.nonzero().flatten().tolist()
        self.width = model.st_hidden_size
        self.count = len(self.sender_indices)
        self.projection = deepcopy(block.gcn.lin)
        self.register_buffer("bias", block.gcn.bias.detach().clone())
        self.input_width = (self.count + 1) * (self.width + 1) + self.count
        self.output_width = self.width

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        end = (self.count + 1) * self.width
        values = packed[..., :end].reshape(*packed.shape[:2], self.count + 1, self.width)
        inverse_degrees = packed[..., end : end + self.count + 1]
        masks = torch.cat([torch.ones_like(packed[..., :1]), packed[..., end + self.count + 1 :]], dim=-1)
        weights = inverse_degrees * inverse_degrees[..., :1] * masks
        return ((self.projection(values) * weights.unsqueeze(-1)).sum(-2) + cast(torch.Tensor, self.bias)).relu()


class STGCNFinishStage(nn.Module):
    def __init__(self, model: STGCNClassifier, block_index: int) -> None:
        super().__init__()
        block = cast(STConvBlock, model.blocks[block_index])
        self.temporal = TemporalProjection(block.temporal2)
        self.width = model.st_hidden_size
        self.original_width = model.features_per_node if block_index == 0 else self.width
        self.residual = nn.Linear(self.original_width, self.width) if isinstance(block.residual, nn.Conv2d) else nn.Identity()
        if isinstance(block.residual, nn.Conv2d):
            self.residual.weight = nn.Parameter(block.residual.weight.detach().clone().squeeze(-1).squeeze(-1))
            self.residual.bias = nn.Parameter(cast(torch.Tensor, block.residual.bias).detach().clone())
        self.input_width = self.original_width + self.width
        self.output_width = self.width

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        return self.temporal(packed[..., self.original_width :]).relu() + self.residual(packed[..., : self.original_width])


class STGCNClassifierStage(nn.Module):
    def __init__(self, model: STGCNClassifier) -> None:
        super().__init__()
        self.classifier = deepcopy(model.fc)
        self.input_width = model.st_hidden_size
        self.output_width = model.num_classes

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.classifier(values)
