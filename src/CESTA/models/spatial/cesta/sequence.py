from __future__ import annotations

import torch


class CESTASequenceMixin:
    crf_transitions: torch.Tensor

    def crf_negative_log_likelihood(
        self,
        emissions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        emissions_seq, targets_seq, mask_seq = self._crf_sequence_tensors(emissions, targets, mask)
        has_valid = mask_seq.any(dim=1)
        if not bool(has_valid.any()):
            return emissions.sum() * 0.0
        emissions_seq = emissions_seq[has_valid]
        targets_seq = targets_seq[has_valid]
        mask_seq = mask_seq[has_valid]
        safe_targets = targets_seq.masked_fill(~mask_seq, 0)
        emissions_seq = emissions_seq.masked_fill(~mask_seq.unsqueeze(-1), 0.0)

        gathered = emissions_seq.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
        transition_scores = self.crf_transitions[safe_targets[:, :-1], safe_targets[:, 1:]]
        transition_mask = mask_seq[:, :-1] & mask_seq[:, 1:]
        sequence_score = gathered.sum(dim=1) + transition_scores.masked_fill(~transition_mask, 0.0).sum(dim=1)

        alpha = torch.zeros_like(emissions_seq[:, 0])
        log_partition = torch.zeros_like(sequence_score)
        for timestep in range(emissions_seq.size(1)):
            current_emission = emissions_seq[:, timestep]
            if timestep == 0:
                alpha = current_emission
            else:
                scores = alpha.unsqueeze(2) + self.crf_transitions.unsqueeze(0) + current_emission.unsqueeze(1)
                next_alpha = torch.logsumexp(scores, dim=1)
                alpha = torch.where(mask_seq[:, timestep - 1].unsqueeze(-1), next_alpha, current_emission)
            segment_end = mask_seq[:, timestep]
            if timestep + 1 < emissions_seq.size(1):
                segment_end = segment_end & ~mask_seq[:, timestep + 1]
            log_partition = log_partition + torch.logsumexp(alpha, dim=1).masked_fill(~segment_end, 0.0)
        return (log_partition - sequence_score).mean()

    @torch.no_grad()
    def crf_decode(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        original_shape = emissions.shape[:-1]
        emissions_seq, _, mask_seq = self._crf_sequence_tensors(emissions, None, mask)
        if emissions_seq.numel() == 0:
            return emissions.argmax(dim=-1)
        emissions_seq = emissions_seq.masked_fill(~mask_seq.unsqueeze(-1), 0.0)
        score = emissions_seq[:, 0]
        backpointers: list[torch.Tensor] = []
        best_labels = [score.argmax(dim=1)]
        for timestep in range(1, emissions_seq.size(1)):
            transition_score = score.unsqueeze(2) + self.crf_transitions.unsqueeze(0)
            best_score, best_path = transition_score.max(dim=1)
            current_emission = emissions_seq[:, timestep]
            score = torch.where(mask_seq[:, timestep - 1].unsqueeze(-1), best_score + current_emission, current_emission)
            backpointers.append(best_path)
            best_labels.append(score.argmax(dim=1))

        decoded = emissions_seq.new_zeros(emissions_seq.shape[:2], dtype=torch.long)
        decoded[:, -1] = best_labels[-1]
        for timestep in range(emissions_seq.size(1) - 2, -1, -1):
            previous = backpointers[timestep].gather(1, decoded[:, timestep + 1].unsqueeze(-1)).squeeze(-1)
            continues = mask_seq[:, timestep] & mask_seq[:, timestep + 1]
            decoded[:, timestep] = torch.where(continues, previous, best_labels[timestep])
        decoded = decoded.masked_fill(~mask_seq, 0)
        if len(original_shape) == 3:
            batch, seq_len, num_nodes = original_shape
            return decoded.view(batch, num_nodes, seq_len).permute(0, 2, 1)
        return decoded.view(*original_shape)

    def _crf_sequence_tensors(
        self,
        emissions: torch.Tensor,
        targets: torch.Tensor | None,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if emissions.ndim == 4:
            batch, seq_len, num_nodes, num_classes = emissions.shape
            emissions_seq = emissions.permute(0, 2, 1, 3).reshape(batch * num_nodes, seq_len, num_classes)
            if targets is None:
                targets_seq = torch.zeros(batch * num_nodes, seq_len, dtype=torch.long, device=emissions.device)
            else:
                targets_seq = targets.permute(0, 2, 1).reshape(batch * num_nodes, seq_len)
            if mask is None:
                mask_seq = (
                    targets_seq >= 0
                    if targets is not None
                    else torch.ones(batch * num_nodes, seq_len, dtype=torch.bool, device=emissions.device)
                )
            else:
                mask_seq = mask.permute(0, 2, 1).reshape(batch * num_nodes, seq_len)
        elif emissions.ndim == 3:
            batch, seq_len, num_classes = emissions.shape
            emissions_seq = emissions.reshape(batch, seq_len, num_classes)
            if targets is None:
                targets_seq = torch.zeros(batch, seq_len, dtype=torch.long, device=emissions.device)
            else:
                targets_seq = targets.reshape(batch, seq_len)
            if mask is None:
                mask_seq = (
                    targets_seq >= 0
                    if targets is not None
                    else torch.ones(batch, seq_len, dtype=torch.bool, device=emissions.device)
                )
            else:
                mask_seq = mask.reshape(batch, seq_len)
        else:
            raise ValueError("CRF emissions must have shape (B, T, C) or (B, T, N, C)")
        if targets is not None:
            mask_seq = mask_seq & (targets_seq >= 0)
        return emissions_seq, targets_seq.long(), mask_seq.bool()
