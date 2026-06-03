"""Main fault injection orchestrator.

Coordinates the full injection pipeline: load data, generate states,
apply faults, return injected DataFrame.
"""

import numpy as np
import pandas as pd
from loguru import logger

from CESTA.datasets.artifact import CESTADataset
from CESTA.datasets.raw.base import BaseDataset
from CESTA.injection.markov import MarkovStateGenerator
from CESTA.injection.registry import get_injector
from CESTA.schema import InjectionConfig


class FaultInjector:
    """Orchestrates the fault injection pipeline.

    Workflow:
        1. Load and preprocess raw dataset
        2. Generate Markov state sequences per group
        3. Apply fault injectors based on states
        4. Return a canonical dataset or transformed DataFrame
    """

    def __init__(self, config: InjectionConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.markov_gen = MarkovStateGenerator(config.markov, self.rng)

    def run(self, dataset: BaseDataset) -> CESTADataset:
        """Execute fault injection and return a canonical dataset without graph edges."""
        df, features = self.run_to_frame(dataset)
        group_col = self.config.group_column
        node_ids = sorted(int(g) for g in df[group_col].unique()) if group_col in df.columns else [0]
        timestamps = sorted(df[dataset.timestamp_column].unique()) if dataset.timestamp_column in df.columns else list(range(len(df)))
        return CESTADataset(
            df=df,
            config=self.config,
            feature_names=features,
            dataset_name=dataset.name,
            timestamp_column=dataset.timestamp_column,
            node_ids=node_ids,
            link_mask=np.empty((len(timestamps), 0), dtype=np.bool_),
        )

    def run_to_frame(self, dataset: BaseDataset) -> tuple[pd.DataFrame, list[str]]:
        df = dataset.load()
        df = dataset.preprocess(
            df,
            resample_freq=self.config.resample_freq,
            interpolation_method=self.config.interpolation_method,
        )
        df, _ = self._inject_faults(df, dataset.group_column)
        features = [f for f in self.config.all_features if f in df.columns]
        return df, features

    def _inject_faults(
        self, df: pd.DataFrame, group_column: str
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Generate fault states and apply fault injectors."""
        df = df.copy()
        all_states = np.zeros(len(df), dtype=np.int32)

        for group_id, group_df in df.groupby(group_column):
            indices = group_df.index.to_numpy()
            length = len(indices)

            states = self.markov_gen.generate(length)
            all_states[indices] = states

            for target_feature in self.config.target_features:
                if target_feature not in df.columns:
                    continue

                data = df.loc[indices, target_feature].to_numpy(dtype=np.float64).copy()

                normal_mask = states == 0
                if np.any(normal_mask):
                    baseline = data[normal_mask]
                else:
                    baseline = data
                mote_std = float(np.std(baseline)) if baseline.size > 1 else 0.0
                mote_median = float(np.median(baseline)) if baseline.size > 0 else 0.0

                for fault_config in self.config.markov.fault_configs:
                    logger.debug(
                        f"Injecting {fault_config.fault_type} into group {group_id}, feature {target_feature}"
                    )
                    fault_type = fault_config.fault_type
                    mask = states == fault_type.value

                    if not np.any(mask):
                        continue

                    enriched_params = dict(fault_config.params)
                    enriched_params.setdefault("_mote_std", mote_std)
                    enriched_params.setdefault("_mote_median", mote_median)

                    injector = get_injector(fault_type)
                    data = injector.apply(data, mask, enriched_params, self.rng)

                df.loc[indices, target_feature] = data

        df["fault_state"] = all_states
        return df, pd.Series(all_states, index=df.index)
