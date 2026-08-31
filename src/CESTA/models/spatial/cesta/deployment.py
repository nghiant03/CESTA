"""Fixed-shape CESTA inference graph for embedded deployment."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from CESTA.models.spatial.cesta.model import CESTAClassifier


class CESTALocalDeploymentModel(nn.Module):
    """Shared local CESTA temporal encoder and classifier for one sensor."""

    def __init__(self, model: CESTAClassifier) -> None:
        super().__init__()
        self.temporal_encoder = model.temporal_encoder
        self.classifier = model.classifier

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return per-timestep probabilities for one sensor window."""
        hidden, _ = self.temporal_encoder(features)
        return F.softmax(self.classifier(hidden), dim=-1)


class CESTANodeDeploymentModel(nn.Module):
    def __init__(self, model: CESTAClassifier, receiver_index: int) -> None:
        super().__init__()
        if not 0 <= receiver_index < model.num_nodes:
            raise ValueError("receiver_index must identify a graph node")
        sender_mask = model.edge_index[1] == receiver_index
        sender_indices = model.edge_index[0, sender_mask]
        if sender_indices.numel() == 0:
            raise ValueError("receiver must have at least one incoming graph edge")
        self.model = model
        self.receiver_index = receiver_index
        self.features_per_node = model.features_per_node
        self.hidden_size = model.encoder_output_size
        self.num_classes = model.num_classes
        self.neighbor_count = int(sender_indices.numel())
        self.input_width = self.features_per_node + self.neighbor_count * (
            self.hidden_size + self.features_per_node
        ) + self.neighbor_count * 2
        self.output_width = self.num_classes + self.hidden_size + self.neighbor_count
        self.register_buffer("sender_indices", sender_indices)
        self.register_buffer("edge_probability", model.edge_prob[receiver_index, sender_indices])

    def forward(self, deployment_input: torch.Tensor) -> torch.Tensor:
        model = self.model
        batch_size, sequence_length, _ = deployment_input.shape
        local_features = deployment_input[..., : self.features_per_node]
        payload_start = self.features_per_node
        payload_end = payload_start + self.neighbor_count * (self.hidden_size + self.features_per_node)
        neighbor_payload = deployment_input[..., payload_start:payload_end].reshape(
            batch_size,
            sequence_length,
            self.neighbor_count,
            self.hidden_size + self.features_per_node,
        )
        neighbor_hidden = neighbor_payload[..., : self.hidden_size]
        neighbor_features = neighbor_payload[..., self.hidden_size :]
        possible_mask = deployment_input[..., payload_end : payload_end + self.neighbor_count]
        received_mask = deployment_input[..., payload_end + self.neighbor_count :]

        local_hidden, _ = model.temporal_encoder(local_features)
        request_mask = self._request_mask(local_hidden, possible_mask)
        neighbor_context = self._aggregate(local_hidden, neighbor_hidden, received_mask)
        fusion_parts = [local_hidden, neighbor_context]
        neighbor_belief_context = None
        if model.use_neighbor_belief:
            neighbor_belief_context = self._masked_neighbor_mean(
                self._belief_features(model.classifier(neighbor_hidden)),
                received_mask,
            )
            fusion_parts.append(neighbor_belief_context)
        fused = model.fusion(torch.cat(fusion_parts, dim=-1))
        hidden = local_hidden + torch.sigmoid(model.graph_residual_logit) * fused
        logits = model.classifier(hidden)
        if model.use_logit_correction:
            correction_delta = self._logit_correction(
                local_hidden,
                neighbor_context,
                local_features,
                neighbor_features,
                received_mask,
                logits,
                neighbor_belief_context,
            )
            if model.use_boundary_gated_correction:
                boundary_gate = 1.0 + torch.sigmoid(model.boundary_head(hidden).squeeze(-1)).unsqueeze(-1)
                correction_delta = boundary_gate * correction_delta
            logits = logits + torch.sigmoid(model.correction_logit) * correction_delta
        probabilities = F.softmax(logits, dim=-1)
        return torch.cat([probabilities, local_hidden, request_mask], dim=-1)

    def _request_mask(self, local_hidden: torch.Tensor, possible_mask: torch.Tensor) -> torch.Tensor:
        model = self.model
        if model.communication_mode == "dense":
            return possible_mask
        if model.communication_mode != "gumbel_request":
            raise ValueError("node export supports dense and gumbel_request communication")
        if model.structured_request_topk > 0:
            raise ValueError("node export does not support structured top-k requests")

        local_belief = self._belief_features(model.classifier(local_hidden))
        entropy = local_belief[..., model.num_classes : model.num_classes + 1]
        margin = local_belief[..., model.num_classes + 1 : model.num_classes + 2]
        receiver_state = local_hidden.unsqueeze(2).expand(-1, -1, self.neighbor_count, -1)
        receiver_entropy = entropy.unsqueeze(2).expand(-1, -1, self.neighbor_count, -1)
        receiver_margin = margin.unsqueeze(2).expand(-1, -1, self.neighbor_count, -1)
        edge_probability = cast(torch.Tensor, self.edge_probability).view(1, 1, self.neighbor_count, 1).expand(
            local_hidden.shape[0],
            local_hidden.shape[1],
            self.neighbor_count,
            1,
        )
        feature_parts = [receiver_state, receiver_entropy, receiver_margin, edge_probability]
        if model.use_temporal_change_gate_feature:
            delta = torch.linalg.vector_norm(local_hidden[:, 1:] - local_hidden[:, :-1], dim=-1, keepdim=True)
            change = torch.cat([delta[:, :1] * 0.0, delta / (1.0 + delta)], dim=1)
            feature_parts.append(change.unsqueeze(2).expand(-1, -1, self.neighbor_count, -1))
        gate_probability = F.softmax(
            model.request_gate(torch.cat(feature_parts, dim=-1) * possible_mask.unsqueeze(-1)),
            dim=-1,
        )[..., 1]
        return gate_probability * possible_mask

    def _aggregate(
        self,
        local_hidden: torch.Tensor,
        neighbor_hidden: torch.Tensor,
        received_mask: torch.Tensor,
    ) -> torch.Tensor:
        model = self.model
        query = model.W_q(local_hidden).unsqueeze(2)
        key = model.W_k(neighbor_hidden)
        value = model.W_v(neighbor_hidden)
        scores = (query * key).sum(dim=-1) / model.attention_scale
        hard_mask = (received_mask != 0).to(scores.dtype)
        has_neighbors = (received_mask.sum(dim=-1, keepdim=True) >= 0.5).to(scores.dtype)
        masked_scores = scores - (1.0 - hard_mask) * 1.0e9
        selected_max = masked_scores.amax(dim=-1, keepdim=True)
        global_max = scores.amax(dim=-1, keepdim=True)
        shift = has_neighbors * selected_max + (1.0 - has_neighbors) * global_max
        shifted_scores = scores - shift
        stable_scores = hard_mask * shifted_scores + (1.0 - hard_mask) * shifted_scores.clamp_max(0.0)
        weights = received_mask * torch.exp(stable_scores)
        denominator = weights.sum(dim=-1, keepdim=True)
        weights = weights / (denominator + (1.0 - has_neighbors))
        return (weights.unsqueeze(-1) * value).sum(dim=2)

    def _belief_features(self, logits: torch.Tensor) -> torch.Tensor:
        probabilities = F.softmax(logits, dim=-1)
        entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-8))).sum(dim=-1, keepdim=True)
        top = probabilities.amax(dim=-1, keepdim=True)
        is_top = (probabilities >= top).to(probabilities.dtype)
        duplicates = (is_top.sum(dim=-1, keepdim=True) >= 1.5).to(probabilities.dtype)
        without_top = (probabilities - is_top * 2.0).amax(dim=-1, keepdim=True).clamp_min(0.0)
        second = duplicates * top + (1.0 - duplicates) * without_top
        margin = top - second
        return torch.cat([probabilities, entropy, margin], dim=-1)

    def _masked_neighbor_mean(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        count = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return (mask.unsqueeze(-1) * values).sum(dim=2) / count

    def _logit_correction(
        self,
        local_hidden: torch.Tensor,
        neighbor_context: torch.Tensor,
        local_features: torch.Tensor,
        neighbor_features: torch.Tensor,
        received_mask: torch.Tensor,
        logits: torch.Tensor,
        neighbor_belief_context: torch.Tensor | None,
    ) -> torch.Tensor:
        model = self.model
        neighbor_feature_context = self._masked_neighbor_mean(neighbor_features, received_mask)
        local_belief = self._belief_features(logits)
        correction_parts = [
            local_hidden,
            neighbor_context,
            local_hidden - neighbor_context,
            local_features,
            neighbor_feature_context,
            local_features - neighbor_feature_context,
            local_belief,
        ]
        if model.use_communication_conditioned_correction:
            correction_parts.extend([local_belief, local_belief * 0.0, local_belief * local_belief])
        if neighbor_belief_context is not None:
            correction_parts.extend(
                [
                    neighbor_belief_context,
                    local_belief - neighbor_belief_context,
                    local_belief * neighbor_belief_context,
                ]
            )
        return model.logit_correction(torch.cat(correction_parts, dim=-1))



class CESTADeploymentModel(nn.Module):
    """Tensor-only CESTA inference graph with static communication metadata."""

    def __init__(self, model: CESTAClassifier) -> None:
        super().__init__()
        self.model = model
        message_mask = torch.zeros(model.num_nodes, model.num_nodes, dtype=torch.float32)
        if model.edge_index.numel() > 0:
            message_mask[model.edge_index[1], model.edge_index[0]] = 1.0
        self.register_buffer("message_mask", message_mask)

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        """Return per-timestep probabilities for every graph node."""
        model = self.model
        batch_size, sequence_length, _, _ = node_features.shape
        local_input = node_features.permute(0, 2, 1, 3).reshape(
            batch_size * model.num_nodes,
            sequence_length,
            model.features_per_node,
        )
        local_hidden, _ = model.temporal_encoder(local_input)
        local_hidden = local_hidden.view(
            batch_size,
            model.num_nodes,
            sequence_length,
            model.encoder_output_size,
        ).permute(0, 2, 1, 3)

        if model.communication_mode == "none":
            hidden = local_hidden
        elif model.communication_mode in {"dense", "gumbel_request"}:
            request_mask = self._request_mask(local_hidden)
            neighbor_context = self._aggregate(local_hidden, request_mask)
            fusion_parts = [local_hidden, neighbor_context]
            neighbor_belief_context = None
            if model.use_neighbor_belief:
                neighbor_belief_context = self._masked_neighbor_mean(
                    self._belief_features(model.classifier(local_hidden)),
                    request_mask,
                )
                fusion_parts.append(neighbor_belief_context)
            fused = model.fusion(torch.cat(fusion_parts, dim=-1))
            hidden = local_hidden + torch.sigmoid(model.graph_residual_logit) * fused
            if model.use_logit_correction:
                logits = model.classifier(hidden)
                correction_delta = self._logit_correction(
                    local_hidden,
                    neighbor_context,
                    node_features,
                    request_mask,
                    logits,
                    neighbor_belief_context,
                )
                if model.use_boundary_gated_correction:
                    boundary_gate = 1.0 + torch.sigmoid(model.boundary_head(hidden).squeeze(-1)).unsqueeze(-1)
                    correction_delta = boundary_gate * correction_delta
                logits = logits + torch.sigmoid(model.correction_logit) * correction_delta
                return F.softmax(logits, dim=-1)
        else:
            raise ValueError("embedded export supports none, dense, and gumbel_request communication")

        return F.softmax(model.classifier(hidden), dim=-1)

    def _request_mask(self, local_hidden: torch.Tensor) -> torch.Tensor:
        model = self.model
        batch_size, sequence_length, _, hidden_size = local_hidden.shape
        possible_mask = cast(torch.Tensor, self.message_mask).view(1, 1, model.num_nodes, model.num_nodes)
        possible_mask = possible_mask.expand(batch_size, sequence_length, model.num_nodes, model.num_nodes)
        if model.communication_mode == "dense":
            return possible_mask
        if model.structured_request_topk > 0:
            raise ValueError("embedded export does not support structured top-k requests")

        local_belief = self._belief_features(model.classifier(local_hidden))
        receiver_state = local_hidden.unsqueeze(3).expand(
            batch_size,
            sequence_length,
            model.num_nodes,
            model.num_nodes,
            hidden_size,
        )
        receiver_entropy = local_belief[..., model.num_classes : model.num_classes + 1].unsqueeze(3)
        receiver_margin = local_belief[..., model.num_classes + 1 : model.num_classes + 2].unsqueeze(3)
        receiver_entropy = receiver_entropy.expand(batch_size, sequence_length, model.num_nodes, model.num_nodes, 1)
        receiver_margin = receiver_margin.expand(batch_size, sequence_length, model.num_nodes, model.num_nodes, 1)
        edge_probability = model.edge_prob.view(1, 1, model.num_nodes, model.num_nodes, 1)
        edge_probability = edge_probability.expand(batch_size, sequence_length, model.num_nodes, model.num_nodes, 1)
        feature_parts = [receiver_state, receiver_entropy, receiver_margin, edge_probability]
        if model.use_temporal_change_gate_feature:
            delta = torch.linalg.vector_norm(local_hidden[:, 1:] - local_hidden[:, :-1], dim=-1, keepdim=True)
            change = torch.cat([delta[:, :1] * 0.0, delta / (1.0 + delta)], dim=1)
            feature_parts.append(change.unsqueeze(3).expand(batch_size, sequence_length, model.num_nodes, model.num_nodes, 1))
        gate_probability = F.softmax(model.request_gate(torch.cat(feature_parts, dim=-1) * possible_mask.unsqueeze(-1)), dim=-1)[..., 1]
        return (gate_probability >= model.request_threshold).to(local_hidden.dtype) * possible_mask

    def _aggregate(self, local_hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        model = self.model
        query = model.W_q(local_hidden)
        key = model.W_k(local_hidden)
        value = model.W_v(local_hidden)
        scores = torch.matmul(query, key.transpose(-1, -2)) / model.attention_scale
        weights = F.softmax(scores - (1.0 - mask) * 1.0e9, dim=-1) * mask
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return torch.matmul(weights, value)

    def _belief_features(self, logits: torch.Tensor) -> torch.Tensor:
        probabilities = F.softmax(logits, dim=-1)
        entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-8))).sum(dim=-1, keepdim=True)
        top = probabilities.amax(dim=-1, keepdim=True)
        is_top = (probabilities >= top).to(probabilities.dtype)
        duplicates = (is_top.sum(dim=-1, keepdim=True) >= 1.5).to(probabilities.dtype)
        without_top = (probabilities - is_top * 2.0).amax(dim=-1, keepdim=True).clamp_min(0.0)
        second = duplicates * top + (1.0 - duplicates) * without_top
        margin = top - second
        return torch.cat([probabilities, entropy, margin], dim=-1)

    def _masked_neighbor_mean(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        count = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return torch.matmul(mask, values) / count

    def _logit_correction(
        self,
        local_hidden: torch.Tensor,
        neighbor_context: torch.Tensor,
        node_features: torch.Tensor,
        mask: torch.Tensor,
        logits: torch.Tensor,
        neighbor_belief_context: torch.Tensor | None,
    ) -> torch.Tensor:
        model = self.model
        neighbor_features = self._masked_neighbor_mean(node_features, mask)
        local_belief = self._belief_features(logits)
        correction_parts = [
            local_hidden,
            neighbor_context,
            local_hidden - neighbor_context,
            node_features,
            neighbor_features,
            node_features - neighbor_features,
            local_belief,
        ]
        if model.use_communication_conditioned_correction:
            communicated_belief = self._belief_features(logits)
            correction_parts.extend(
                [
                    communicated_belief,
                    communicated_belief - local_belief,
                    communicated_belief * local_belief,
                ]
            )
        if neighbor_belief_context is not None:
            correction_parts.extend(
                [
                    neighbor_belief_context,
                    local_belief - neighbor_belief_context,
                    local_belief * neighbor_belief_context,
                ]
            )
        return model.logit_correction(torch.cat(correction_parts, dim=-1))
