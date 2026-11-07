from __future__ import annotations

import numpy as np
import pandas as pd

from CESTA.datasets.raw.base import BaseDataset
from CESTA.schema.fault import FaultType


class CestaDHTDataset(BaseDataset):
    device_nodes = {"esp32_00": 1, "esp32_01": 2, "esp32_02": 3, "esp32_03": 4}

    @property
    def name(self) -> str:
        return "cesta_dht"

    @property
    def feature_columns(self) -> list[str]:
        return ["temp", "humid"]

    @property
    def group_column(self) -> str:
        return "moteid"

    @property
    def timestamp_column(self) -> str:
        return "timestamp"

    def load(self) -> pd.DataFrame:
        path = self.data_path if self.data_path.is_file() else self.data_path / "readings.jsonl"
        frame = pd.read_json(path, lines=True, convert_dates=False)
        required = {"device_id", "timestamp", "temperature", "humidity", "path", "fault_mode"}
        if frame.empty or not required.issubset(frame.columns):
            raise ValueError(f"Empty DHT recording or missing fields: {sorted(required - set(frame.columns))}")
        if frame[list(required)].isna().any().any():
            raise ValueError("DHT recording contains missing required values")
        if not set(frame.device_id).issubset(self.device_nodes):
            raise ValueError("Unknown DHT device ID; update the explicit device-to-node mapping")
        labels = {fault.name.lower(): fault.value for fault in FaultType}
        if not set(frame.fault_mode).issubset(labels) or not set(frame.path).issubset({"normal", "fault"}):
            raise ValueError("Unknown DHT path or fault mode")
        if ((frame.path == "normal") != (frame.fault_mode == "normal")).any():
            raise ValueError("DHT path and fault mode disagree")
        if frame.duplicated(["device_id", "path", "timestamp"]).any():
            raise ValueError("Duplicate DHT device/path/timestamp readings")
        numeric = frame[["timestamp", "temperature", "humidity"]].to_numpy(dtype=np.float64)
        if not np.isfinite(numeric).all() or (numeric[:, 0] % 1 != 0).any():
            raise ValueError("DHT readings require finite values and integer Unix-second timestamps")
        frame = frame.rename(columns={"temperature": "temp", "humidity": "humid"})
        frame["timestamp"] = pd.to_datetime(frame.timestamp, unit="s", utc=True)
        frame["moteid"] = frame.device_id.map(self.device_nodes).astype(int)
        frame["fault_state"] = frame.fault_mode.map(labels).astype(np.int32)
        return frame.sort_values(["moteid", "timestamp", "path"]).reset_index(drop=True)

    def preprocess(
        self,
        df: pd.DataFrame,
        resample_freq: str = "3s",
        interpolation_method: str = "none",
    ) -> pd.DataFrame:
        if pd.Timedelta(resample_freq) != pd.Timedelta(seconds=3) or interpolation_method != "none":
            raise ValueError("Pre-labeled DHT ingestion requires resample_freq='3s' and interpolation_method='none'")
        seconds = df.timestamp.astype("int64") // 1_000_000_000
        if ((seconds - seconds.min()) % 3 != 0).any():
            raise ValueError("DHT timestamps do not align to the shared three-second sampling grid")
        unique_seconds = np.sort(seconds.unique())
        if (np.diff(unique_seconds) != 3).any():
            raise ValueError("Recording contains a whole-network gap; ingest contiguous sessions separately")
        fault_devices = set(df.loc[df.path == "fault", "device_id"])
        selected = df.loc[(df.path == "fault") | ~df.device_id.isin(fault_devices)].copy()
        reference = df.loc[df.path == "normal", ["device_id", "timestamp", "temp", "humid"]].rename(
            columns={"temp": "reference_temp", "humid": "reference_humid"}
        )
        selected = selected.merge(reference, on=["device_id", "timestamp"], how="left", validate="one_to_one")
        if selected[["reference_temp", "reference_humid"]].isna().any().any():
            raise ValueError("Selected DHT readings lack their paired clean reference")
        return selected.sort_values(["moteid", "timestamp"]).reset_index(drop=True)
