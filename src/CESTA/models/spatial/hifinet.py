"""HiFiNet temporal-graph architecture for per-node fault classification."""

from __future__ import annotations

from typing import ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from CESTA.batch import GraphWindowBatch
from CESTA.models.base import BaseModel


class HiFiNetClassifier(BaseModel):
    """LSTM and graph-attention model with per-timestep node predictions."""

    required_metadata: ClassVar[set[str]] = {"graph"}

    def __init__(
        self,
        input_size: int,
        num_nodes: int,
        edge_index: list[list[int]] | None = None,
        temporal_hidden_size: int = 128,
        embedding_size: int = 64,
        gat_hidden_size: int = 64,
        gat_heads: int = 8,
        num_gat_layers: int = 2,
        classifier_hidden_sizes: list[int] | None = None,
        num_classes: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if input_size % num_nodes != 0:
            raise ValueError("input_size must be divisible by num_nodes")
        if gat_hidden_size % gat_heads != 0:
            raise ValueError("gat_hidden_size must be divisible by gat_heads")
        if num_gat_layers < 1:
            raise ValueError("num_gat_layers must be at least 1")

        self.input_size = input_size
        self.num_nodes = num_nodes
        self.features_per_node = input_size // num_nodes
        self.temporal_hidden_size = temporal_hidden_size
        self.embedding_size = embedding_size
        self.gat_hidden_size = gat_hidden_size
        self.gat_heads = gat_heads
        self.num_gat_layers = num_gat_layers
        self.classifier_hidden_sizes = classifier_hidden_sizes or [128, 64]
        self.num_classes = num_classes
        self.dropout_probability = dropout

        if edge_index is None:
            edge_index_tensor = torch.arange(num_nodes, dtype=torch.long).repeat(2, 1)
        else:
            edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
        if edge_index_tensor.ndim != 2 or edge_index_tensor.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, num_edges)")
        self.register_buffer("edge_index", edge_index_tensor)
        self._edge_index_list = edge_index_tensor.tolist()

        self.temporal_encoder_1 = nn.LSTM(
            input_size=self.features_per_node,
            hidden_size=temporal_hidden_size,
            batch_first=True,
        )
        self.temporal_encoder_2 = nn.LSTM(
            input_size=temporal_hidden_size,
            hidden_size=embedding_size,
            batch_first=True,
        )

        channels_per_head = gat_hidden_size // gat_heads
        self.graph_layers = nn.ModuleList()
        self.graph_norms = nn.ModuleList()
        graph_input_size = embedding_size
        for _ in range(num_gat_layers):
            self.graph_layers.append(
                GATConv(
                    graph_input_size,
                    channels_per_head,
                    heads=gat_heads,
                    concat=True,
                    dropout=dropout,
                )
            )
            self.graph_norms.append(nn.LayerNorm(gat_hidden_size))
            graph_input_size = gat_hidden_size

        classifier_layers: list[nn.Module] = []
        classifier_input_size = embedding_size + gat_hidden_size
        for hidden_size in self.classifier_hidden_sizes:
            classifier_layers.extend(
                [
                    nn.Linear(classifier_input_size, hidden_size),
                    nn.LayerNorm(hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            classifier_input_size = hidden_size
        classifier_layers.append(nn.Linear(classifier_input_size, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

    @property
    def name(self) -> str:
        return "hifinet"

    def forward(self, x: torch.Tensor | GraphWindowBatch) -> torch.Tensor:
        edge_mask: torch.Tensor | None = None
        edge_index: torch.Tensor = self.edge_index  # type: ignore[assignment]
        if isinstance(x, GraphWindowBatch):
            edge_index = x.edge_index
            edge_mask = x.edge_mask
            x = x.x

        if x.ndim == 4:
            batch_size, sequence_length, _, _ = x.shape
            node_inputs = x
        else:
            batch_size, sequence_length, _ = x.shape
            node_inputs = x.view(batch_size, sequence_length, self.num_nodes, self.features_per_node)

        temporal_inputs = node_inputs.permute(0, 2, 1, 3).reshape(
            batch_size * self.num_nodes,
            sequence_length,
            self.features_per_node,
        )
        temporal_features, _ = self.temporal_encoder_1(temporal_inputs)
        temporal_features, _ = self.temporal_encoder_2(temporal_features)
        temporal_features = temporal_features.view(batch_size, self.num_nodes, sequence_length, self.embedding_size)
        temporal_features = temporal_features.permute(0, 2, 1, 3)

        graph_features = temporal_features.reshape(batch_size * sequence_length * self.num_nodes, self.embedding_size)
        batched_edge_index = self._batch_edge_index(
            edge_index,
            edge_mask,
            batch_size=batch_size,
            sequence_length=sequence_length,
            device=graph_features.device,
        )
        for graph_layer, graph_norm in zip(self.graph_layers, self.graph_norms, strict=True):
            graph_features = graph_layer(graph_features, batched_edge_index)
            graph_features = F.relu(graph_norm(graph_features))

        graph_features = graph_features.view(batch_size, sequence_length, self.num_nodes, self.gat_hidden_size)
        combined_features = torch.cat([temporal_features, graph_features], dim=-1)
        return self.classifier(combined_features)

    def _batch_edge_index(
        self,
        edge_index: torch.Tensor,
        edge_mask: torch.Tensor | None,
        *,
        batch_size: int,
        sequence_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        edge_index = edge_index.to(device)
        graph_count = batch_size * sequence_length
        offsets = torch.arange(graph_count, device=device) * self.num_nodes
        if edge_mask is None:
            return (edge_index.unsqueeze(0) + offsets[:, None, None]).permute(1, 0, 2).reshape(2, -1)

        active_edges = edge_mask.reshape(graph_count, -1).to(device)
        edge_parts = [edge_index[:, active_edges[index]] + offsets[index] for index in range(graph_count) if bool(active_edges[index].any())]
        if not edge_parts:
            return torch.empty((2, 0), dtype=torch.long, device=device)
        return torch.cat(edge_parts, dim=1)

    def get_config(self) -> dict[str, object]:
        return {
            "input_size": self.input_size,
            "num_nodes": self.num_nodes,
            "edge_index": self._edge_index_list,
            "temporal_hidden_size": self.temporal_hidden_size,
            "embedding_size": self.embedding_size,
            "gat_hidden_size": self.gat_hidden_size,
            "gat_heads": self.gat_heads,
            "num_gat_layers": self.num_gat_layers,
            "classifier_hidden_sizes": self.classifier_hidden_sizes,
            "num_classes": self.num_classes,
            "dropout": self.dropout_probability,
        }
