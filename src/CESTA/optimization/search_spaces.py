"""Per-model Optuna search spaces.

Each search space is a callable ``(trial) -> dict`` that returns the
``model_kwargs`` to pass to ``create_model``. Common training-hyperparameter
suggestions live in :func:`suggest_train_hyperparams`.

New models can register their search space with :func:`register_search_space`.
"""

from __future__ import annotations

from typing import Any, Callable

import optuna

SearchSpaceFn = Callable[[optuna.trial.Trial], dict[str, Any]]


def suggest_train_hyperparams(trial: optuna.trial.Trial) -> dict[str, Any]:
    """Suggest model-agnostic training hyperparameters.

    Tunes optimizer learning rate, batch size, focal-loss usage, and
    minority-oversampling ratio.
    """
    use_focal = trial.suggest_categorical("use_focal_loss", [False, True])
    overrides: dict[str, Any] = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
        "use_focal_loss": use_focal,
    }
    if use_focal:
        overrides["focal_gamma"] = trial.suggest_float("focal_gamma", 0.5, 4.0)

    oversample = trial.suggest_categorical("oversample", [False, True])
    overrides["oversample"] = oversample
    if oversample:
        overrides["oversample_ratio"] = trial.suggest_float(
            "oversample_ratio", 0.1, 1.0
        )
    return overrides


def _cnn1d_space(trial: optuna.trial.Trial) -> dict[str, Any]:
    return {
        "num_channels": trial.suggest_categorical("num_channels", [32, 64, 128]),
        "num_blocks": trial.suggest_int("num_blocks", 2, 6),
        "kernel_size": trial.suggest_categorical("kernel_size", [3, 5, 7]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "dilation_base": trial.suggest_categorical("dilation_base", [1, 2]),
    }


def _transformer_space(trial: optuna.trial.Trial) -> dict[str, Any]:
    n_heads = trial.suggest_categorical("n_heads", [2, 4, 8])
    d_model = trial.suggest_categorical("d_model", [32, 64, 128])
    if d_model % n_heads != 0:
        raise optuna.TrialPruned(
            f"d_model={d_model} not divisible by n_heads={n_heads}"
        )
    return {
        "d_model": d_model,
        "num_layers": trial.suggest_int("num_layers", 1, 4),
        "n_heads": n_heads,
        "d_ff": trial.suggest_categorical("d_ff", [64, 128, 256]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.4),
    }


def _autoformer_space(trial: optuna.trial.Trial) -> dict[str, Any]:
    space = _transformer_space(trial)
    space["moving_average"] = trial.suggest_categorical("moving_average", [3, 5, 7, 9])
    return space


def _informer_space(trial: optuna.trial.Trial) -> dict[str, Any]:
    space = _transformer_space(trial)
    space["sampling_factor"] = trial.suggest_int("sampling_factor", 3, 7)
    return space


def _patchtst_space(trial: optuna.trial.Trial) -> dict[str, Any]:
    space = _transformer_space(trial)
    space["patch_length"] = trial.suggest_categorical("patch_length", [4, 8, 16])
    space["patch_stride"] = trial.suggest_categorical("patch_stride", [1, 2, 4])
    return space


def _modern_tcn_space(trial: optuna.trial.Trial) -> dict[str, Any]:
    return {
        "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128]),
        "num_blocks": trial.suggest_int("num_blocks", 2, 6),
        "kernel_size": trial.suggest_categorical("kernel_size", [7, 11, 15, 19]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.4),
        "expansion_ratio": trial.suggest_float("expansion_ratio", 1.0, 4.0),
        "dilation_base": trial.suggest_categorical("dilation_base", [1, 2]),
    }


def _hydra_space(trial: optuna.trial.Trial) -> dict[str, Any]:
    return {
        "d_model": trial.suggest_categorical("d_model", [32, 64, 128]),
        "num_layers": trial.suggest_int("num_layers", 1, 3),
        "d_state": trial.suggest_categorical("d_state", [32, 64, 128]),
        "d_conv": trial.suggest_categorical("d_conv", [3, 5, 7]),
        "expand": trial.suggest_categorical("expand", [1, 2]),
        "head_dim": trial.suggest_categorical("head_dim", [16, 32]),
        "num_groups": 1,
        "dropout": trial.suggest_float("dropout", 0.0, 0.4),
    }


def _stgcn_space(trial: optuna.trial.Trial) -> dict[str, Any]:
    return {
        "st_hidden": trial.suggest_categorical("st_hidden", [32, 64, 128]),
        "num_st_blocks": trial.suggest_int("num_st_blocks", 1, 3),
        "temporal_kernel": trial.suggest_categorical("temporal_kernel", [3, 5, 7]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.4),
    }


def _hifinet_space(trial: optuna.trial.Trial) -> dict[str, Any]:
    return {
        "temporal_hidden_size": trial.suggest_categorical("temporal_hidden_size", [64, 128, 256]),
        "embedding_size": trial.suggest_categorical("embedding_size", [32, 64, 128]),
        "gat_hidden_size": trial.suggest_categorical("gat_hidden_size", [32, 64, 128]),
        "gat_heads": trial.suggest_categorical("gat_heads", [1, 2, 4, 8]),
        "num_gat_layers": trial.suggest_int("num_gat_layers", 1, 3),
        "dropout": trial.suggest_float("dropout", 0.0, 0.4),
    }


def _dcrnn_space(trial: optuna.trial.Trial) -> dict[str, Any]:
    return {
        "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128]),
        "num_layers": trial.suggest_int("num_layers", 1, 3),
        "diffusion_steps": trial.suggest_int("diffusion_steps", 1, 3),
        "bidirectional_diffusion": trial.suggest_categorical("bidirectional_diffusion", [False, True]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.4),
    }


_REGISTRY: dict[str, SearchSpaceFn] = {
    "cnn1d": _cnn1d_space,
    "transformer": _transformer_space,
    "autoformer": _autoformer_space,
    "informer": _informer_space,
    "patchtst": _patchtst_space,
    "modern_tcn": _modern_tcn_space,
    "hydra": _hydra_space,
    "stgcn": _stgcn_space,
    "hifinet": _hifinet_space,
    "dcrnn": _dcrnn_space,
}


def register_search_space(model_name: str, fn: SearchSpaceFn) -> None:
    """Register a search-space function for a model architecture."""
    _REGISTRY[model_name] = fn


def get_search_space(model_name: str) -> SearchSpaceFn:
    """Look up a model's search-space function.

    Raises:
        KeyError: if no search space is registered for the model.
    """
    if model_name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise KeyError(
            f"No search space registered for model '{model_name}'. "
            f"Available: {available}"
        )
    return _REGISTRY[model_name]
