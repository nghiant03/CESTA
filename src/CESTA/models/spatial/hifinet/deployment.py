from __future__ import annotations

from copy import deepcopy
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv

from CESTA.models.spatial.hifinet import HiFiNetClassifier


class UnrolledLSTM(nn.Module):
    def __init__(self, layer: nn.LSTM) -> None:
        super().__init__()
        if layer.num_layers != 1 or layer.bidirectional or layer.proj_size or not layer.bias or not layer.batch_first:
            raise ValueError("deployment requires a single-layer forward LSTM with bias and batch-first input")
        self.hidden_size = layer.hidden_size
        for name in ("weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0"):
            self.register_buffer(name, getattr(layer, name).detach().clone())

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = features.new_zeros((features.shape[0], self.hidden_size))
        cell = torch.zeros_like(hidden)
        outputs = []
        for step in range(features.shape[1]):
            gates = F.linear(features[:, step], cast(torch.Tensor, self.weight_ih_l0), cast(torch.Tensor, self.bias_ih_l0))
            gates = gates + F.linear(hidden, cast(torch.Tensor, self.weight_hh_l0), cast(torch.Tensor, self.bias_hh_l0))
            incoming, forget, candidate, outgoing = gates.chunk(4, dim=-1)
            cell = forget.sigmoid() * cell + incoming.sigmoid() * candidate.tanh()
            hidden = outgoing.sigmoid() * cell.tanh()
            outputs.append(hidden.unsqueeze(1))
        return torch.cat(outputs, dim=1)


class HiFiNetTemporalStage(nn.Module):
    def __init__(self, model: HiFiNetClassifier) -> None:
        super().__init__()
        self.first = UnrolledLSTM(model.temporal_encoder_1)
        self.second = UnrolledLSTM(model.temporal_encoder_2)
        self.input_width = model.features_per_node
        self.output_width = model.embedding_size

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.second(self.first(features))


class HiFiNetGraphStage(nn.Module):
    def __init__(self, model: HiFiNetClassifier, receiver_index: int, layer_index: int) -> None:
        super().__init__()
        if not 0 <= receiver_index < model.num_nodes or not 0 <= layer_index < model.num_gat_layers:
            raise ValueError("invalid receiver or graph layer index")
        edges = cast(torch.Tensor, model.edge_index)
        if edges.numel() and (int(edges.min()) < 0 or int(edges.max()) >= model.num_nodes):
            raise ValueError("graph contains invalid node indices")
        selected = (edges[1] == receiver_index) & (edges[0] != receiver_index)
        self.sender_indices = edges[0, selected].tolist()
        self.edge_indices = selected.nonzero().flatten().tolist()
        layer = cast(GATConv, model.graph_layers[layer_index])
        if layer.lin is None or not layer.concat or not layer.add_self_loops or layer.res is not None or layer.edge_dim is not None:
            raise ValueError("unsupported GAT configuration")
        self.projection = deepcopy(layer.lin)
        self.register_buffer("attention_source", layer.att_src.detach().clone())
        self.register_buffer("attention_target", layer.att_dst.detach().clone())
        self.register_buffer("bias", layer.bias.detach().clone())
        self.norm = deepcopy(model.graph_norms[layer_index])
        self.heads = layer.heads
        self.channels = layer.out_channels
        self.negative_slope = layer.negative_slope
        self.payload_width = model.embedding_size if layer_index == 0 else model.gat_hidden_size
        self.neighbor_count = len(self.sender_indices)
        self.input_width = (1 + self.neighbor_count) * self.payload_width + self.neighbor_count
        self.output_width = model.gat_hidden_size

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        batch, time, _ = packed.shape
        end = (1 + self.neighbor_count) * self.payload_width
        values = packed[..., :end].reshape(batch, time, 1 + self.neighbor_count, self.payload_width)
        projected = self.projection(values).reshape(batch, time, 1 + self.neighbor_count, self.heads, self.channels)
        source = (projected * cast(torch.Tensor, self.attention_source)).sum(-1)
        target = (projected[:, :, :1] * cast(torch.Tensor, self.attention_target)).sum(-1)
        scores = source + target
        scores = scores.clamp_min(0.0) + self.negative_slope * scores.clamp_max(0.0)
        mask = torch.cat([torch.ones_like(packed[..., :1]), (packed[..., end:] != 0).to(packed.dtype)], dim=-1).unsqueeze(-1)
        scores = torch.where(mask != 0, scores, torch.full_like(scores, -1.0e30))
        weights = torch.softmax(scores, dim=2) * mask
        aggregated = (projected * weights.unsqueeze(-1)).sum(2).reshape(batch, time, self.output_width)
        return F.relu(self.norm(aggregated + cast(torch.Tensor, self.bias)))


class HiFiNetClassifierStage(nn.Module):
    def __init__(self, model: HiFiNetClassifier) -> None:
        super().__init__()
        self.classifier = deepcopy(model.classifier).eval()
        self.input_width = model.embedding_size + model.gat_hidden_size
        self.output_width = model.num_classes

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        return self.classifier(packed)
