"""Diffusion convolutional recurrent network for per-node fault classification."""

from __future__ import annotations

from typing import ClassVar

import torch
import torch.nn as nn

from CESTA.batch import GraphWindowBatch
from CESTA.models.base import BaseModel


class DiffusionGraphConv(nn.Module):
    """Graph diffusion over directed random-walk supports."""

    def __init__(self, input_size: int, output_size: int, diffusion_steps: int, support_count: int) -> None:
        super().__init__()
        self.diffusion_steps = diffusion_steps
        self.support_count = support_count
        expanded_input_size = input_size * (1 + diffusion_steps * support_count)
        self.projection = nn.Linear(expanded_input_size, output_size)

    def forward(self, x: torch.Tensor, supports: tuple[torch.Tensor, ...]) -> torch.Tensor:
        features = [x]
        for support in supports:
            diffused = x
            for _ in range(self.diffusion_steps):
                diffused = torch.bmm(support, diffused)
                features.append(diffused)
        return self.projection(torch.cat(features, dim=-1))


class DCGRUCell(nn.Module):
    """GRU cell whose gates and candidate use graph diffusion."""

    def __init__(self, input_size: int, hidden_size: int, diffusion_steps: int, support_count: int) -> None:
        super().__init__()
        combined_size = input_size + hidden_size
        self.hidden_size = hidden_size
        self.gates = DiffusionGraphConv(combined_size, hidden_size * 2, diffusion_steps, support_count)
        self.candidate = DiffusionGraphConv(combined_size, hidden_size, diffusion_steps, support_count)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor, supports: tuple[torch.Tensor, ...]) -> torch.Tensor:
        combined = torch.cat([x, hidden], dim=-1)
        reset, update = torch.sigmoid(self.gates(combined, supports)).chunk(2, dim=-1)
        candidate_input = torch.cat([x, reset * hidden], dim=-1)
        candidate = torch.tanh(self.candidate(candidate_input, supports))
        return update * hidden + (1.0 - update) * candidate


class DCRNNClassifier(BaseModel):
    """DCRNN baseline with dynamic graph masks and per-timestep node logits."""

    required_metadata: ClassVar[set[str]] = {"graph"}

    def __init__(
        self,
        input_size: int,
        num_nodes: int,
        edge_index: list[list[int]] | None = None,
        hidden_size: int = 64,
        num_layers: int = 2,
        diffusion_steps: int = 2,
        bidirectional_diffusion: bool = True,
        num_classes: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if input_size % num_nodes != 0:
            raise ValueError("input_size must be divisible by num_nodes")
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if diffusion_steps < 1:
            raise ValueError("diffusion_steps must be at least 1")

        self.input_size = input_size
        self.num_nodes = num_nodes
        self.features_per_node = input_size // num_nodes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.diffusion_steps = diffusion_steps
        self.bidirectional_diffusion = bidirectional_diffusion
        self.num_classes = num_classes
        self.dropout_probability = dropout

        if edge_index is None:
            edge_index_tensor = torch.arange(num_nodes, dtype=torch.long).repeat(2, 1)
        else:
            edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
        if edge_index_tensor.ndim != 2 or edge_index_tensor.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, num_edges)")
        self.register_buffer("edge_index", edge_index_tensor)
        self._edge_index_list: list[list[int]] = edge_index_tensor.tolist()

        support_count = 2 if bidirectional_diffusion else 1
        cells: list[DCGRUCell] = []
        layer_input_size = self.features_per_node
        for _ in range(num_layers):
            cells.append(DCGRUCell(layer_input_size, hidden_size, diffusion_steps, support_count))
            layer_input_size = hidden_size
        self.cells = nn.ModuleList(cells)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    @property
    def name(self) -> str:
        return "dcrnn"

    def forward(self, x: torch.Tensor | GraphWindowBatch) -> torch.Tensor:
        edge_mask: torch.Tensor | None = None
        edge_index: torch.Tensor = self.edge_index  # type: ignore[assignment]
        if isinstance(x, GraphWindowBatch):
            edge_index = x.edge_index
            edge_mask = x.edge_mask
            x = x.x

        if x.ndim == 4:
            node_inputs = x
        else:
            batch_size, sequence_length, _ = x.shape
            node_inputs = x.view(batch_size, sequence_length, self.num_nodes, self.features_per_node)

        batch_size, sequence_length, _, _ = node_inputs.shape
        hidden_states = [node_inputs.new_zeros(batch_size, self.num_nodes, self.hidden_size) for _ in self.cells]
        outputs: list[torch.Tensor] = []

        for timestep in range(sequence_length):
            timestep_mask = None if edge_mask is None else edge_mask[:, timestep]
            supports = self._build_supports(edge_index, timestep_mask, batch_size, node_inputs.device, node_inputs.dtype)
            layer_input = node_inputs[:, timestep]
            for layer_index, cell in enumerate(self.cells):
                hidden_states[layer_index] = cell(layer_input, hidden_states[layer_index], supports)
                layer_input = hidden_states[layer_index]
                if layer_index + 1 < self.num_layers:
                    layer_input = self.dropout(layer_input)
            outputs.append(self.classifier(layer_input))

        return torch.stack(outputs, dim=1)

    def _build_supports(
        self,
        edge_index: torch.Tensor,
        edge_mask: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, ...]:
        edge_index = edge_index.to(device)
        edge_count = edge_index.shape[1]
        if edge_mask is None:
            active = torch.ones((batch_size, edge_count), dtype=dtype, device=device)
        else:
            active = edge_mask.to(device=device, dtype=dtype)

        adjacency = torch.zeros((batch_size, self.num_nodes, self.num_nodes), dtype=dtype, device=device)
        source, receiver = edge_index
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, edge_count)
        receiver_indices = receiver.unsqueeze(0).expand(batch_size, -1)
        source_indices = source.unsqueeze(0).expand(batch_size, -1)
        adjacency.index_put_((batch_indices, receiver_indices, source_indices), active, accumulate=True)
        forward = self._row_normalize(adjacency)
        if not self.bidirectional_diffusion:
            return (forward,)
        return forward, self._row_normalize(adjacency.transpose(1, 2))

    @staticmethod
    def _row_normalize(adjacency: torch.Tensor) -> torch.Tensor:
        degree = adjacency.sum(dim=-1, keepdim=True)
        return adjacency / degree.clamp_min(1.0)

    def get_config(self) -> dict[str, object]:
        return {
            "input_size": self.input_size,
            "num_nodes": self.num_nodes,
            "edge_index": self._edge_index_list,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "diffusion_steps": self.diffusion_steps,
            "bidirectional_diffusion": self.bidirectional_diffusion,
            "num_classes": self.num_classes,
            "dropout": self.dropout_probability,
        }
