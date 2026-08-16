"""Hydra bidirectional state-space model for per-timestep fault classification."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from CESTA.models.base import BaseModel


class HydraMixer(nn.Module):
    """Portable Hydra quasiseparable matrix mixer."""

    def __init__(
        self,
        d_model: int,
        *,
        d_state: int = 64,
        d_conv: int = 7,
        expand: int = 2,
        head_dim: int = 64,
        num_groups: int = 1,
    ) -> None:
        super().__init__()
        if d_state < 1:
            msg = "d_state must be >= 1"
            raise ValueError(msg)
        if d_conv < 1 or d_conv % 2 == 0:
            msg = "d_conv must be a positive odd integer"
            raise ValueError(msg)
        if expand < 1:
            msg = "expand must be >= 1"
            raise ValueError(msg)
        if head_dim < 1:
            msg = "head_dim must be >= 1"
            raise ValueError(msg)
        if num_groups < 1:
            msg = "num_groups must be >= 1"
            raise ValueError(msg)

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.head_dim = head_dim
        self.num_groups = num_groups
        self.inner_size = expand * d_model
        if self.inner_size % head_dim != 0:
            msg = "expand * d_model must be divisible by head_dim"
            raise ValueError(msg)
        self.num_heads = self.inner_size // head_dim
        if self.num_heads % num_groups != 0:
            msg = "the number of heads must be divisible by num_groups"
            raise ValueError(msg)

        projection_size = 2 * self.inner_size + 4 * num_groups * d_state + 2 * self.num_heads
        convolution_size = self.inner_size + 4 * num_groups * d_state
        self.input_projection = nn.Linear(d_model, projection_size, bias=False)
        self.convolution = nn.Conv1d(
            convolution_size,
            convolution_size,
            kernel_size=d_conv,
            padding=d_conv // 2,
            groups=convolution_size,
        )
        self.log_decay = nn.Parameter(torch.zeros(self.num_heads))
        initial_step = torch.exp(torch.rand(self.num_heads) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        self.step_bias = nn.Parameter(initial_step + torch.log(-torch.expm1(-initial_step)))
        self.diagonal_bias = nn.Parameter(torch.ones(self.num_heads))
        self.diagonal_projection = nn.Linear(self.inner_size, self.num_heads, bias=False)
        self.norm_weight = nn.Parameter(torch.ones(self.inner_size))
        self.output_projection = nn.Linear(self.inner_size, d_model, bias=False)

    def _scan(
        self,
        values: torch.Tensor,
        steps: torch.Tensor,
        input_states: torch.Tensor,
        output_states: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sequence_length, _, _ = values.shape
        heads_per_group = self.num_heads // self.num_groups
        input_states = input_states.repeat_interleave(heads_per_group, dim=2)
        output_states = output_states.repeat_interleave(heads_per_group, dim=2)
        decay_rates = -torch.exp(self.log_decay.float()).to(dtype=values.dtype)
        state = values.new_zeros(batch_size, self.num_heads, self.head_dim, self.d_state)
        outputs: list[torch.Tensor] = []
        for timestep in range(sequence_length):
            step = steps[:, timestep]
            decay = torch.exp(step * decay_rates.unsqueeze(0))
            state = state * decay[:, :, None, None]
            state = state + values[:, timestep, :, :, None] * step[:, :, None, None] * input_states[:, timestep, :, None, :]
            outputs.append(torch.einsum("bhpn,bhn->bhp", state, output_states[:, timestep]))
        mixed = torch.stack(outputs, dim=1)
        return torch.cat((torch.zeros_like(mixed[:, :1]), mixed[:, :-1]), dim=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Mix a ``(batch, time, d_model)`` sequence in both directions."""
        projected = self.input_projection(inputs)
        state_width = 4 * self.num_groups * self.d_state
        gate, convolution_input, raw_steps = torch.split(
            projected,
            [self.inner_size, self.inner_size + state_width, 2 * self.num_heads],
            dim=-1,
        )
        convolution_output = F.silu(self.convolution(convolution_input.transpose(1, 2)).transpose(1, 2))
        values, bidirectional_states = torch.split(convolution_output, [self.inner_size, state_width], dim=-1)
        forward_states, backward_states = torch.chunk(bidirectional_states, 2, dim=-1)
        forward_input, forward_output = torch.chunk(forward_states, 2, dim=-1)
        backward_input, backward_output = torch.chunk(backward_states, 2, dim=-1)
        state_shape = (*forward_input.shape[:-1], self.num_groups, self.d_state)
        forward_input = forward_input.view(state_shape)
        forward_output = forward_output.view(state_shape)
        backward_input = backward_input.view(state_shape)
        backward_output = backward_output.view(state_shape)
        forward_steps, backward_steps = torch.chunk(raw_steps, 2, dim=-1)
        forward_steps = F.softplus(forward_steps + self.step_bias)
        backward_steps = F.softplus(backward_steps.flip(1) + self.step_bias)

        head_values = values.view(*values.shape[:-1], self.num_heads, self.head_dim)
        forward = self._scan(head_values, forward_steps, forward_input, forward_output)
        backward = self._scan(
            head_values.flip(1),
            backward_steps,
            backward_input.flip(1),
            backward_output.flip(1),
        ).flip(1)
        diagonal_scale = self.diagonal_projection(values) + self.diagonal_bias
        diagonal = head_values * diagonal_scale.unsqueeze(-1)
        mixed = (forward + backward + diagonal).flatten(start_dim=-2)
        variance = mixed.square().mean(dim=-1, keepdim=True)
        mixed = mixed * torch.rsqrt(variance + 1e-5) * self.norm_weight
        mixed = mixed * F.silu(gate)
        return self.output_projection(mixed)


class HydraLayer(nn.Module):
    """Residual Hydra mixer layer."""

    def __init__(
        self,
        d_model: int,
        *,
        d_state: int,
        d_conv: int,
        expand: int,
        head_dim: int,
        num_groups: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.mixer = HydraMixer(
            d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            head_dim=head_dim,
            num_groups=num_groups,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply bidirectional mixing, residual dropout, and normalization."""
        return self.norm(inputs + self.dropout(self.mixer(inputs)))


class HydraClassifier(BaseModel):
    """Hydra classifier with one output label for every input timestep."""

    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        num_layers: int = 2,
        num_classes: int = 4,
        d_state: int = 64,
        d_conv: int = 7,
        expand: int = 2,
        head_dim: int = 64,
        num_groups: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            msg = "num_layers must be >= 1"
            raise ValueError(msg)

        self.input_size = input_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.head_dim = head_dim
        self.num_groups = num_groups
        self.dropout_prob = dropout
        self.input_projection = nn.Linear(input_size, d_model)
        self.layers = nn.ModuleList(
            [
                HydraLayer(
                    d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    head_dim=head_dim,
                    num_groups=num_groups,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    @property
    def name(self) -> str:
        return "hydra"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits with shape ``(batch, time, num_classes)``."""
        hidden = self.input_projection(x)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.classifier(self.output_dropout(hidden))

    def get_config(self) -> dict[str, object]:
        """Return model configuration for artifact serialization."""
        return {
            "input_size": self.input_size,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "num_classes": self.num_classes,
            "d_state": self.d_state,
            "d_conv": self.d_conv,
            "expand": self.expand,
            "head_dim": self.head_dim,
            "num_groups": self.num_groups,
            "dropout": self.dropout_prob,
        }
