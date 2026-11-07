from __future__ import annotations

import math

from torch.optim import Optimizer

from CESTA.schema.config import TrainConfig


def epoch_learning_rate(config: TrainConfig, epoch: int) -> float:
    if not 1 <= epoch <= config.epochs:
        raise ValueError("Epoch outside configured training schedule")
    if config.learning_rate_schedule == "none":
        return config.learning_rate
    start = config.learning_rate_schedule_start_epoch
    final = config.learning_rate_final
    if start is None or final is None:
        raise ValueError("Incomplete learning-rate schedule")
    if epoch <= start:
        return config.learning_rate
    fraction = (epoch - start) / (config.epochs - start)
    return final + (config.learning_rate - final) * 0.5 * (1.0 + math.cos(math.pi * fraction))


def apply_epoch_learning_rate(optimizer: Optimizer, config: TrainConfig, epoch: int) -> list[float]:
    if config.learning_rate_schedule != "none":
        rate = epoch_learning_rate(config, epoch)
        for group in optimizer.param_groups:
            group["lr"] = rate
    return [float(group["lr"]) for group in optimizer.param_groups]
