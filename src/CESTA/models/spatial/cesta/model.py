"""CESTA spatial-temporal model for communication-aware fault diagnosis."""

from __future__ import annotations

import math
from typing import ClassVar, Literal

import torch
import torch.nn as nn

from CESTA.batch import GraphWindowBatch
from CESTA.models.base import BaseModel
from CESTA.models.spatial.cesta.communication import CESTACommunicationMixin, CommunicationStats
from CESTA.models.spatial.cesta.sequence import CESTASequenceMixin

CommunicationMode = Literal[
    "none",
    "dense",
    "gumbel_request",
    "random",
    "static_topk",
    "local_change",
]

_RULE_COMMUNICATION_MODES = {"random", "static_topk", "local_change"}


class CESTAClassifier(CESTASequenceMixin, CESTACommunicationMixin, BaseModel):
    """Communication-Efficient Spatial-Temporal Aggregation classifier.

    Uses GAT-inspired single-head attention for neighbor aggregation.
    When zero neighbors are requested, produces a zero context vector.
    """

    required_metadata: ClassVar[set[str]] = {"graph"}

    def __init__(
        self,
        input_size: int,
        num_nodes: int,
        edge_index: list[list[int]] | None = None,
        edge_prob: list[float] | None = None,
        edge_distance_m: list[float] | None = None,
        hidden_size: int = 64,
        num_layers: int = 1,
        num_classes: int = 4,
        dropout: float = 0.2,
        communication_mode: CommunicationMode = "none",
        fusion_hidden_size: int | None = None,
        precision_bits: int = 32,
        gumbel_temperature: float = 1.0,
        request_threshold: float = 0.5,
        gate_hidden_size: int = 32,
        use_temporal_change_gate_feature: bool = False,
        num_attention_heads: int = 1,
        graph_residual_init: float = 1.0,
        bidirectional: bool = False,
        use_logit_correction: bool = False,
        correction_hidden_size: int | None = None,
        correction_init: float = 0.1,
        use_neighbor_belief: bool = False,
        use_boundary_head: bool = False,
        boundary_hidden_size: int | None = None,
        use_boundary_gated_correction: bool = False,
        use_crf: bool = False,
        use_communication_conditioned_correction: bool = False,
        structured_request_topk: int = 0,
        control_request_ratio: float = 0.3,
        control_seed: int = 42,
        control_static_topk: int = 1,
        control_local_change_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        if input_size % num_nodes != 0:
            raise ValueError("input_size must be divisible by num_nodes")
        allowed_communication_modes = {"none", "dense", "gumbel_request", *_RULE_COMMUNICATION_MODES}
        if communication_mode not in allowed_communication_modes:
            raise ValueError(f"communication_mode must be one of: {', '.join(sorted(allowed_communication_modes))}")
        if gumbel_temperature <= 0.0:
            raise ValueError("gumbel_temperature must be positive")
        if not 0.0 <= request_threshold <= 1.0:
            raise ValueError("request_threshold must be in [0, 1]")
        if gate_hidden_size < 1:
            raise ValueError("gate_hidden_size must be positive")
        if num_attention_heads < 1:
            raise ValueError("num_attention_heads must be positive")
        if not 0.0 <= graph_residual_init <= 1.0:
            raise ValueError("graph_residual_init must be in [0, 1]")
        if correction_hidden_size is not None and correction_hidden_size < 1:
            raise ValueError("correction_hidden_size must be positive")
        if not 0.0 <= correction_init <= 1.0:
            raise ValueError("correction_init must be in [0, 1]")
        if boundary_hidden_size is not None and boundary_hidden_size < 1:
            raise ValueError("boundary_hidden_size must be positive")
        if use_boundary_gated_correction and not use_boundary_head:
            raise ValueError("use_boundary_gated_correction requires use_boundary_head")
        if structured_request_topk < 0:
            raise ValueError("structured_request_topk must be non-negative")
        if not 0.0 <= control_request_ratio <= 1.0:
            raise ValueError("control_request_ratio must be in [0, 1]")
        if control_static_topk < 1:
            raise ValueError("control_static_topk must be positive")
        if not 0.0 <= control_local_change_threshold <= 1.0:
            raise ValueError("control_local_change_threshold must be in [0, 1]")

        self.input_size = input_size
        self.num_nodes = num_nodes
        self.features_per_node = input_size // num_nodes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout_prob = dropout
        self.communication_mode: CommunicationMode = communication_mode
        self.fusion_hidden_size = fusion_hidden_size
        self.precision_bits = precision_bits
        self.gumbel_temperature = gumbel_temperature
        self.request_threshold = request_threshold
        self.gate_hidden_size = gate_hidden_size
        self.use_temporal_change_gate_feature = use_temporal_change_gate_feature
        self.num_attention_heads = num_attention_heads
        self.graph_residual_init = graph_residual_init
        self.bidirectional = bidirectional
        self.use_logit_correction = use_logit_correction
        self.correction_hidden_size = correction_hidden_size
        self.correction_init = correction_init
        self.use_neighbor_belief = use_neighbor_belief
        self.use_boundary_head = use_boundary_head
        self.boundary_hidden_size = boundary_hidden_size
        self.use_boundary_gated_correction = use_boundary_gated_correction
        self.use_crf = use_crf
        self.use_communication_conditioned_correction = use_communication_conditioned_correction
        self.structured_request_topk = structured_request_topk
        self.control_request_ratio = control_request_ratio
        self.control_seed = control_seed
        self.control_static_topk = control_static_topk
        self.control_local_change_threshold = control_local_change_threshold
        self._active_window_ids: torch.Tensor | None = None
        self.encoder_output_size = hidden_size * (2 if bidirectional else 1)
        self.neighbor_belief_size = num_classes + 2

        self._gate_entropy: torch.Tensor | None = None
        self._last_boundary_logits: torch.Tensor | None = None
        self._local_logits: torch.Tensor | None = None
        self._communication_logits: torch.Tensor | None = None
        self._communication_activity: torch.Tensor | None = None
        self._receiver_request_probability: torch.Tensor | None = None

        if num_attention_heads != 1:
            raise NotImplementedError(
                "Multi-head attention (>1) is not yet implemented"
            )
        if self.encoder_output_size % num_attention_heads != 0:
            raise ValueError("encoder output size must be divisible by num_attention_heads")

        if edge_index is not None:
            edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
        else:
            edge_index_tensor = torch.empty((2, 0), dtype=torch.long)
        if edge_index_tensor.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, num_edges)")
        if edge_prob is not None:
            edge_prob_tensor = torch.tensor(edge_prob, dtype=torch.float32)
        else:
            edge_prob_tensor = torch.ones((edge_index_tensor.shape[1],), dtype=torch.float32)
        if edge_prob_tensor.shape != (edge_index_tensor.shape[1],):
            raise ValueError("edge_prob must have shape (num_edges,)")
        if edge_distance_m is not None:
            edge_distance_tensor = torch.tensor(edge_distance_m, dtype=torch.float32)
        else:
            edge_distance_tensor = torch.zeros((edge_index_tensor.shape[1],), dtype=torch.float32)
        if edge_distance_tensor.shape != (edge_index_tensor.shape[1],):
            raise ValueError("edge_distance_m must have shape (num_edges,)")
        if bool((edge_distance_tensor < 0.0).any()):
            raise ValueError("edge_distance_m must be non-negative")
        self.register_buffer("edge_index", edge_index_tensor)
        self.register_buffer("edge_prob_values", edge_prob_tensor)
        self.register_buffer("edge_distance_m", edge_distance_tensor, persistent=False)
        edge_prob_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
        if edge_index_tensor.numel() > 0:
            edge_prob_matrix[edge_index_tensor[1], edge_index_tensor[0]] = edge_prob_tensor
        self.register_buffer("edge_prob", edge_prob_matrix)
        self._edge_index_list: list[list[int]] = edge_index_tensor.tolist()
        self._edge_prob_list: list[float] = edge_prob_tensor.tolist()

        self.temporal_encoder = nn.GRU(
            input_size=self.features_per_node,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout)
        self.attention_scale = self.encoder_output_size ** 0.5
        self.W_q = nn.Linear(self.encoder_output_size, self.encoder_output_size, bias=False)
        self.W_k = nn.Linear(self.encoder_output_size, self.encoder_output_size, bias=False)
        self.W_v = nn.Linear(self.encoder_output_size, self.encoder_output_size, bias=False)
        fusion_output_size = fusion_hidden_size or self.encoder_output_size
        fusion_input_size = self.encoder_output_size * 2 + (self.neighbor_belief_size if use_neighbor_belief else 0)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_size, fusion_output_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_output_size, self.encoder_output_size),
        )
        residual_eps = 1e-4
        residual_init = min(max(graph_residual_init, residual_eps), 1.0 - residual_eps)
        self.graph_residual_logit = nn.Parameter(
            torch.tensor(math.log(residual_init / (1.0 - residual_init)), dtype=torch.float32)
        )
        self.gate_input_schema = ["receiver_hidden", "receiver_entropy", "receiver_margin", "edge_probability"]
        if use_temporal_change_gate_feature:
            self.gate_input_schema.append("receiver_temporal_change")
        request_gate_input_size = self.encoder_output_size + len(self.gate_input_schema) - 1
        self.request_gate = nn.Sequential(
            nn.Linear(request_gate_input_size, gate_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_size, 2),
        )
        self.need_gate = nn.Sequential(
            nn.Linear(self.encoder_output_size + 2, gate_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_size, 2),
        )
        self.neighbor_ranker = nn.Sequential(
            nn.Linear(self.encoder_output_size + 3, gate_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_size, 1),
        )
        self.classifier = nn.Linear(self.encoder_output_size, num_classes)
        boundary_layer_size = boundary_hidden_size or self.encoder_output_size
        self.boundary_head = nn.Sequential(
            nn.Linear(self.encoder_output_size, boundary_layer_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(boundary_layer_size, 1),
        )
        correction_input_size = self.encoder_output_size * 3 + self.features_per_node * 3 + num_classes + 2
        if use_communication_conditioned_correction:
            correction_input_size += self.neighbor_belief_size * 3
        if use_neighbor_belief:
            correction_input_size += self.neighbor_belief_size * 3
        correction_layer_size = correction_hidden_size or self.encoder_output_size
        self.logit_correction = nn.Sequential(
            nn.Linear(correction_input_size, correction_layer_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(correction_layer_size, num_classes),
        )
        correction_eps = 1e-4
        correction_scale_init = min(max(correction_init, correction_eps), 1.0 - correction_eps)
        self.correction_logit = nn.Parameter(
            torch.tensor(math.log(correction_scale_init / (1.0 - correction_scale_init)), dtype=torch.float32)
        )
        self.crf_transitions = nn.Parameter(torch.zeros(num_classes, num_classes))
        self._freeze_inactive_parameters()
        self._last_communication_stats: CommunicationStats = (
            self._zero_communication_stats()
        )
        self._communication_loss: torch.Tensor | None = None
        self._communication_energy_loss: torch.Tensor | None = None
        self._gate_entropy: torch.Tensor | None = None

    def _freeze_inactive_parameters(self) -> None:
        inactive_modules: list[nn.Module] = []
        if self.communication_mode != "gumbel_request":
            inactive_modules.extend([self.request_gate, self.need_gate, self.neighbor_ranker])
        if not self.use_boundary_head:
            inactive_modules.append(self.boundary_head)
        if not self.use_logit_correction:
            inactive_modules.append(self.logit_correction)
        for module in inactive_modules:
            for parameter in module.parameters():
                parameter.requires_grad = False
        if not self.use_boundary_head:
            self.crf_transitions.requires_grad = self.use_crf
        if not self.use_logit_correction:
            self.correction_logit.requires_grad = False

    @property
    def name(self) -> str:
        return "cesta"

    @property
    def last_communication_stats(self) -> CommunicationStats:
        return self._last_communication_stats.copy()

    @property
    def auxiliary_loss(self) -> torch.Tensor | None:
        """Communication loss for backward compatibility."""
        return self._communication_loss

    @property
    def communication_loss(self) -> torch.Tensor | None:
        return self._communication_loss

    @property
    def communication_energy_loss(self) -> torch.Tensor | None:
        return self._communication_energy_loss

    @property
    def graph_residual_scale(self) -> torch.Tensor:
        return torch.sigmoid(self.graph_residual_logit)

    @property
    def correction_scale(self) -> torch.Tensor:
        return torch.sigmoid(self.correction_logit)

    @property
    def gate_entropy(self) -> torch.Tensor | None:
        return self._gate_entropy

    @property
    def local_logits(self) -> torch.Tensor | None:
        return self._local_logits

    @property
    def communication_logits(self) -> torch.Tensor | None:
        return self._communication_logits

    @property
    def communication_activity(self) -> torch.Tensor | None:
        return self._communication_activity

    @property
    def receiver_request_probability(self) -> torch.Tensor | None:
        return self._receiver_request_probability

    @property
    def last_boundary_logits(self) -> torch.Tensor | None:
        return self._last_boundary_logits

    def set_gumbel_temperature(self, tau: float) -> None:
        """Update Gumbel-Softmax temperature for annealing."""
        if tau <= 0.0:
            raise ValueError("gumbel_temperature must be positive")
        self.gumbel_temperature = tau

    def forward(self, x: torch.Tensor | GraphWindowBatch) -> torch.Tensor:
        edge_index: torch.Tensor | None = None
        edge_mask: torch.Tensor | None = None
        if isinstance(x, GraphWindowBatch):
            edge_index = x.edge_index
            edge_mask = x.edge_mask
            self._active_window_ids = x.window_ids
            x = x.x
        else:
            self._active_window_ids = None

        if x.ndim == 4:
            batch, seq_len, _, _ = x.shape
            node_features = x
        else:
            batch, seq_len, _ = x.shape
            node_features = x.view(batch, seq_len, self.num_nodes, self.features_per_node)
        local_input = node_features.permute(0, 2, 1, 3).reshape(
            batch * self.num_nodes, seq_len, self.features_per_node
        )

        local_hidden, _ = self.temporal_encoder(local_input)
        local_hidden = local_hidden.view(
            batch, self.num_nodes, seq_len, self.encoder_output_size
        )
        local_hidden = local_hidden.permute(0, 2, 1, 3)

        correction_context: tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None = None
        if self.communication_mode == "dense":
            neighbor_context, possible_mask = self._dense_neighbor_context(
                local_hidden, edge_index=edge_index, edge_mask=edge_mask
            )
            neighbor_belief_context = self._neighbor_belief_context(local_hidden, possible_mask) if self.use_neighbor_belief else None
            fusion_input = [local_hidden, neighbor_context]
            if neighbor_belief_context is not None:
                fusion_input.append(neighbor_belief_context)
            fused = self.fusion(torch.cat(fusion_input, dim=-1))
            hidden = self.dropout(local_hidden + self.graph_residual_scale * fused)
            correction_context = (neighbor_context, possible_mask, neighbor_belief_context)
            self._last_communication_stats = self._dense_communication_stats(
                possible_mask=possible_mask,
                edge_index=edge_index,
            )
            self._communication_loss = self._dense_communication_loss(possible_mask)
            self._communication_energy_loss = self._expected_energy_ratio(possible_mask, possible_mask, edge_index=edge_index)
            self._communication_activity = self._receiver_request_activity(possible_mask, possible_mask)
            self._receiver_request_probability = self._communication_activity
            self._gate_entropy = None
        elif self.communication_mode == "gumbel_request":
            neighbor_context, request_mask, possible_mask, soft_gate_probs = (
                self._gumbel_neighbor_context(
                    local_hidden, edge_index=edge_index, edge_mask=edge_mask
                )
            )
            neighbor_belief_context = self._neighbor_belief_context(local_hidden, request_mask) if self.use_neighbor_belief else None
            fusion_input = [local_hidden, neighbor_context]
            if neighbor_belief_context is not None:
                fusion_input.append(neighbor_belief_context)
            fused = self.fusion(torch.cat(fusion_input, dim=-1))
            hidden = self.dropout(local_hidden + self.graph_residual_scale * fused)
            correction_context = (neighbor_context, request_mask, neighbor_belief_context)
            self._last_communication_stats = self._request_communication_stats(
                request_mask=request_mask,
                possible_mask=possible_mask,
                edge_index=edge_index,
            )
            self._communication_loss = self._request_communication_loss(
                request_mask=request_mask,
                possible_mask=possible_mask,
            )
            self._communication_energy_loss = self._expected_energy_ratio(
                soft_gate_probs[..., 1], possible_mask, edge_index=edge_index
            )
            self._communication_activity = self._receiver_request_activity(request_mask, possible_mask)
            self._receiver_request_probability = self._receiver_request_activity(soft_gate_probs[..., 1], possible_mask)
            self._gate_entropy = self._compute_gate_entropy(soft_gate_probs, possible_mask)
        elif self.communication_mode in _RULE_COMMUNICATION_MODES:
            neighbor_context, request_mask, possible_mask = self._rule_neighbor_context(
                local_hidden, edge_index=edge_index, edge_mask=edge_mask
            )
            neighbor_belief_context = self._neighbor_belief_context(local_hidden, request_mask) if self.use_neighbor_belief else None
            fusion_input = [local_hidden, neighbor_context]
            if neighbor_belief_context is not None:
                fusion_input.append(neighbor_belief_context)
            fused = self.fusion(torch.cat(fusion_input, dim=-1))
            hidden = self.dropout(local_hidden + self.graph_residual_scale * fused)
            correction_context = (neighbor_context, request_mask, neighbor_belief_context)
            self._last_communication_stats = self._request_communication_stats(
                request_mask=request_mask,
                possible_mask=possible_mask,
                edge_index=edge_index,
            )
            self._communication_loss = self._request_communication_loss(
                request_mask=request_mask,
                possible_mask=possible_mask,
            )
            self._communication_energy_loss = self._expected_energy_ratio(request_mask, possible_mask, edge_index=edge_index)
            self._communication_activity = self._receiver_request_activity(request_mask, possible_mask)
            self._receiver_request_probability = self._communication_activity
            self._gate_entropy = None
        else:
            possible_mask = self._possible_message_mask(local_hidden, edge_index=edge_index, edge_mask=edge_mask)
            hidden = self.dropout(local_hidden)
            self._last_communication_stats = self._zero_communication_stats(possible_mask=possible_mask, edge_index=edge_index)
            self._communication_loss = torch.zeros((), dtype=local_hidden.dtype, device=x.device)
            self._communication_energy_loss = torch.zeros((), dtype=local_hidden.dtype, device=x.device)
            self._communication_activity = torch.zeros(batch, seq_len, self.num_nodes, dtype=local_hidden.dtype, device=x.device)
            self._receiver_request_probability = self._communication_activity
            self._gate_entropy = None

        self._last_boundary_logits = self.boundary_head(hidden).squeeze(-1) if self.use_boundary_head else None
        local_logits = self.classifier(local_hidden)
        logits = self.classifier(hidden)
        self._local_logits = local_logits
        self._communication_logits = logits if correction_context is not None else None
        if self.use_logit_correction and correction_context is not None:
            neighbor_context, correction_mask, neighbor_belief_context = correction_context
            communicated_logits = self._communication_logits if self.use_communication_conditioned_correction else None
            correction_delta = self._logit_correction(
                local_hidden=local_hidden,
                neighbor_context=neighbor_context,
                node_features=node_features,
                mask=correction_mask,
                local_logits=logits,
                communicated_logits=communicated_logits,
                neighbor_belief_context=neighbor_belief_context,
            )
            if self.use_boundary_gated_correction and self._last_boundary_logits is not None:
                boundary_gate = 1.0 + torch.sigmoid(self._last_boundary_logits).unsqueeze(-1)
                correction_delta = boundary_gate * correction_delta
            logits = logits + self.correction_scale * correction_delta
        return logits

    def get_config(self) -> dict[str, object]:
        return {
            "input_size": self.input_size,
            "num_nodes": self.num_nodes,
            "edge_index": self._edge_index_list,
            "edge_prob": self._edge_prob_list,
            "edge_distance_m": self.edge_distance_m.tolist(),
            "graph_edge_count": len(self._edge_prob_list),
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_classes": self.num_classes,
            "dropout": self.dropout_prob,
            "communication_mode": self.communication_mode,
            "fusion_hidden_size": self.fusion_hidden_size,
            "precision_bits": self.precision_bits,
            "gumbel_temperature": self.gumbel_temperature,
            "request_threshold": self.request_threshold,
            "gate_hidden_size": self.gate_hidden_size,
            "use_temporal_change_gate_feature": self.use_temporal_change_gate_feature,
            "gate_input_schema": self.gate_input_schema,
            "num_attention_heads": self.num_attention_heads,
            "graph_residual_init": self.graph_residual_init,
            "bidirectional": self.bidirectional,
            "use_logit_correction": self.use_logit_correction,
            "correction_hidden_size": self.correction_hidden_size,
            "correction_init": self.correction_init,
            "use_neighbor_belief": self.use_neighbor_belief,
            "use_boundary_head": self.use_boundary_head,
            "boundary_hidden_size": self.boundary_hidden_size,
            "use_boundary_gated_correction": self.use_boundary_gated_correction,
            "use_crf": self.use_crf,
            "use_communication_conditioned_correction": self.use_communication_conditioned_correction,
            "structured_request_topk": self.structured_request_topk,
            "control_request_ratio": self.control_request_ratio,
            "control_seed": self.control_seed,
            "control_static_topk": self.control_static_topk,
            "control_local_change_threshold": self.control_local_change_threshold,
        }

