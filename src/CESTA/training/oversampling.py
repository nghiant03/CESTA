"""Oversampling utilities for imbalanced datasets.

Provides window-level oversampling of minority (non-NORMAL) classes
by duplicating windows that contain at least one non-normal label.
"""

import numpy as np
from numpy.typing import NDArray

from CESTA.logging import logger
from CESTA.schema.fault import FaultType


def oversample_minority(
    X: NDArray[np.float32],
    y: NDArray[np.int32],
    ratio: float = 1.0,
    seed: int | None = None,
    node_mask: NDArray[np.bool_] | None = None,
    edge_mask: NDArray[np.bool_] | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.int32], NDArray[np.bool_] | None, NDArray[np.bool_] | None]:
    """Oversample windows containing non-NORMAL labels."""
    normal_val = FaultType.NORMAL.value

    flat_per_window = y.reshape(len(y), -1)
    is_minority = np.any(flat_per_window != normal_val, axis=1)
    minority_idx = np.where(is_minority)[0]
    majority_idx = np.where(~is_minority)[0]

    n_minority = len(minority_idx)
    n_majority = len(majority_idx)

    if n_minority == 0:
        logger.warning("No minority samples found, skipping oversampling")
        return X, y, node_mask, edge_mask

    target_minority = int(n_majority * ratio)
    n_to_add = max(0, target_minority - n_minority)

    if n_to_add == 0:
        logger.info("Minority already meets target ratio, no oversampling needed")
        return X, y, node_mask, edge_mask

    rng = np.random.default_rng(seed)
    extra_idx = rng.choice(minority_idx, size=n_to_add, replace=True)

    X_out = np.concatenate([X, X[extra_idx]], axis=0)
    y_out = np.concatenate([y, y[extra_idx]], axis=0)
    node_mask_out = np.concatenate([node_mask, node_mask[extra_idx]], axis=0) if node_mask is not None else None
    edge_mask_out = np.concatenate([edge_mask, edge_mask[extra_idx]], axis=0) if edge_mask is not None else None

    shuffle = rng.permutation(len(X_out))
    X_out = X_out[shuffle]
    y_out = y_out[shuffle]
    node_mask_out = node_mask_out[shuffle] if node_mask_out is not None else None
    edge_mask_out = edge_mask_out[shuffle] if edge_mask_out is not None else None

    logger.info(
        "Oversampled: {} -> {} windows (added {} minority copies)",
        len(X),
        len(X_out),
        n_to_add,
    )

    return X_out, y_out, node_mask_out, edge_mask_out
