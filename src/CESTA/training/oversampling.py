"""Split-aware oversampling utilities for imbalanced datasets."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from CESTA.datasets.injected.windowed import WindowedSplit
from CESTA.logging import logger


def oversample_split(
    split: WindowedSplit,
    ratio: float = 1.0,
    seed: int | None = None,
    normal_class_id: int = 0,
) -> WindowedSplit:
    """Oversample windows containing labels other than ``normal_class_id``."""
    y = split.y
    flat_per_window = y.reshape(len(y), -1)
    is_minority = np.any(flat_per_window != normal_class_id, axis=1)
    minority_idx = np.where(is_minority)[0]
    majority_idx = np.where(~is_minority)[0]

    n_minority = len(minority_idx)
    n_majority = len(majority_idx)

    if n_minority == 0:
        logger.warning("No minority samples found, skipping oversampling")
        return split

    target_minority = int(n_majority * ratio)
    n_to_add = max(0, target_minority - n_minority)

    if n_to_add == 0:
        logger.info("Minority already meets target ratio, no oversampling needed")
        return split

    rng = np.random.default_rng(seed)
    extra_idx = rng.choice(minority_idx, size=n_to_add, replace=True)
    selected = np.concatenate([np.arange(len(y)), extra_idx])
    shuffle = rng.permutation(len(selected))
    selected = selected[shuffle]

    sampled = split.select(selected)
    logger.info(
        "Oversampled: {} -> {} windows (added {} minority copies)",
        len(split.X),
        len(sampled.X),
        n_to_add,
    )
    return sampled


def oversample_minority(
    X: NDArray[np.float32],
    y: NDArray[np.int32],
    ratio: float = 1.0,
    seed: int | None = None,
    node_mask: NDArray[np.bool_] | None = None,
    edge_mask: NDArray[np.bool_] | None = None,
    normal_class_id: int = 0,
) -> tuple[NDArray[np.float32], NDArray[np.int32], NDArray[np.bool_] | None, NDArray[np.bool_] | None]:
    """Compatibility wrapper around split-aware oversampling."""
    split = oversample_split(
        WindowedSplit(X=X, y=y, node_mask=node_mask, edge_mask=edge_mask),
        ratio=ratio,
        seed=seed,
        normal_class_id=normal_class_id,
    )
    return split.X, split.y, split.node_mask, split.edge_mask
