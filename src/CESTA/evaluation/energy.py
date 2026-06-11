"""First-order radio-energy accounting for communication metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class RadioEnergyConfig:
    electronics_j_per_bit: float = 50e-9
    free_space_j_per_bit_m2: float = 10e-12
    multipath_j_per_bit_m4: float = 0.0013e-12

    @property
    def crossover_distance_m(self) -> float:
        return math.sqrt(self.free_space_j_per_bit_m2 / self.multipath_j_per_bit_m4)

    def to_payload(self) -> dict[str, Any]:
        return {
            "radio_model": "first_order_tx_rx",
            "electronics_j_per_bit": self.electronics_j_per_bit,
            "free_space_j_per_bit_m2": self.free_space_j_per_bit_m2,
            "multipath_j_per_bit_m4": self.multipath_j_per_bit_m4,
            "crossover_distance_m": self.crossover_distance_m,
            "tx_equation": "bits * (E_elec + E_fs*d^2) for d < d0 else bits * (E_elec + E_mp*d^4)",
            "rx_equation": "bits * E_elec",
        }


def compute_radio_energy_metrics(
    *,
    requested_edge_counts: Sequence[float],
    possible_edge_counts: Sequence[float],
    edge_distance_m: Sequence[float],
    bits_per_message: float,
    distance_metadata: dict[str, Any] | None = None,
    config: RadioEnergyConfig | None = None,
) -> dict[str, Any]:
    constants = config or RadioEnergyConfig()
    requested = _as_nonnegative_array(requested_edge_counts, "requested_edge_counts")
    possible = _as_nonnegative_array(possible_edge_counts, "possible_edge_counts")
    distances = _as_nonnegative_array(edge_distance_m, "edge_distance_m")
    if requested.shape != possible.shape or requested.shape != distances.shape:
        msg = (
            "Energy accounting requires requested counts, possible counts, "
            f"and distances to have matching shapes; got {requested.shape}, {possible.shape}, {distances.shape}"
        )
        raise ValueError(msg)
    if bits_per_message < 0.0:
        raise ValueError("bits_per_message must be non-negative")

    selective = _energy_totals(requested, distances, bits_per_message, constants)
    dense = _energy_totals(possible, distances, bits_per_message, constants)
    dense_total = dense["total_energy_j"]
    dense_messages = dense["message_count"]
    dense_bits = dense["bit_count"]
    return {
        "constants": constants.to_payload(),
        "units": {
            "energy": "J",
            "distance": "m",
            "payload": "bit",
        },
        "distance": _distance_payload(distance_metadata, edge_count=int(distances.size)),
        "selective": selective,
        "dense_reference": dense,
        "dense_vs_selective": {
            "selective_subset_of_dense": bool(np.all(requested <= possible + 1e-6)),
            "total_energy_reduction_j": dense_total - selective["total_energy_j"],
            "total_energy_reduction_ratio": (dense_total - selective["total_energy_j"]) / dense_total if dense_total > 0.0 else 0.0,
            "message_count_reduction_ratio": (dense_messages - selective["message_count"]) / dense_messages if dense_messages > 0.0 else 0.0,
            "transmitted_bit_reduction_ratio": (dense_bits - selective["bit_count"]) / dense_bits if dense_bits > 0.0 else 0.0,
        },
    }


def _as_nonnegative_array(values: Sequence[float], name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if np.any(array < 0.0):
        raise ValueError(f"{name} must be non-negative")
    return array


def _energy_totals(
    edge_counts: NDArray[np.float64],
    distances: NDArray[np.float64],
    bits_per_message: float,
    config: RadioEnergyConfig,
) -> dict[str, float]:
    bit_counts = edge_counts * bits_per_message
    free_space = distances < config.crossover_distance_m
    amplifier_j_per_bit = np.where(
        free_space,
        config.free_space_j_per_bit_m2 * np.square(distances),
        config.multipath_j_per_bit_m4 * np.power(distances, 4),
    )
    tx_energy_by_edge = bit_counts * (config.electronics_j_per_bit + amplifier_j_per_bit)
    rx_energy_by_edge = bit_counts * config.electronics_j_per_bit
    tx_energy = float(tx_energy_by_edge.sum())
    rx_energy = float(rx_energy_by_edge.sum())
    total_energy = tx_energy + rx_energy
    message_count = float(edge_counts.sum())
    return {
        "edge_count": float(edge_counts.size),
        "message_count": message_count,
        "bit_count": float(bit_counts.sum()),
        "bits_per_message": float(bits_per_message),
        "tx_energy_j": tx_energy,
        "rx_energy_j": rx_energy,
        "total_energy_j": total_energy,
        "tx_share": tx_energy / total_energy if total_energy > 0.0 else 0.0,
        "rx_share": rx_energy / total_energy if total_energy > 0.0 else 0.0,
        "free_space_message_count": float(edge_counts[free_space].sum()),
        "multipath_message_count": float(edge_counts[~free_space].sum()),
        "weighted_mean_distance_m": float(np.average(distances, weights=edge_counts)) if message_count > 0.0 else 0.0,
        "max_distance_m": float(distances.max()) if distances.size > 0 else 0.0,
    }


def _distance_payload(distance_metadata: dict[str, Any] | None, *, edge_count: int) -> dict[str, Any]:
    payload = dict(distance_metadata or {})
    payload.setdefault("units", "m")
    payload["edge_count"] = edge_count
    return payload
