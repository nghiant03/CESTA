//! Static node configuration.

#![allow(dead_code)]

use crate::fault::{FaultConfig, FaultMode};

pub const WIFI_SSID: &str = "Wifi Guest";
pub const WIFI_PASSWORD: &str = "2222288888";

pub const MQTT_SERVER: &str = "192.168.0.145";
pub const MQTT_PORT: u16 = 1883;
pub const MQTT_USER: &str = "";
pub const MQTT_PASSWORD: &str = "";

pub const DEVICE_ID: &str = "esp32_01";
pub const MQTT_TOPIC_PREFIX: &str = "cesta/readings/";

pub const NODE_INDEX: usize = 0;

pub const NEIGHBORS: [Neighbor; 2] = [
    Neighbor {
        device_id: "esp32_Y",
        node_index: 2,
        mac: [0x00; 6],
    },
    Neighbor {
        device_id: "esp32_Z",
        node_index: 30,
        mac: [0x00; 6],
    },
];

pub const EXCHANGE_WAIT_MS: u64 = 1500;

pub const EXCHANGE_POLL_MS: u64 = 50;

pub const MQTT_BUFFER_BYTES: usize = 8 * 1024;

pub const DHT_PIN: i32 = 5;
pub const SEND_INTERVAL_SECS: u64 = 3;

pub const INFERENCE_ENABLED: bool = true;
pub const INFERENCE_SYNTHETIC_DIAGNOSTIC: bool = false;
pub const INFERENCE_TENSOR_ARENA_BYTES: usize = 2 * 1024 * 1024;

pub const NTP_SERVER: &str = "vn.pool.ntp.org";
pub const NTP_SYNC_TIMEOUT_SECS: u64 = 5;
pub const NTP_SYNC_POLL_MS: u64 = 500;

pub const FAULT_CONFIG: FaultConfig = FAULT_NORMAL;

pub const NORMAL_TEMPERATURE_SIGMA: [f32; 4] = [
    0.991493173924548,
    1.339996615837971,
    1.4351699959638797,
    1.2784088857507183,
];

pub const BASELINE_SIGMA: f32 = {
    assert!(
        NODE_INDEX < NORMAL_TEMPERATURE_SIGMA.len(),
        "Calibration requires dataset nodes 1..4 mapped to NODE_INDEX 0..3"
    );
    NORMAL_TEMPERATURE_SIGMA[NODE_INDEX]
};

pub const FAULT_NORMAL: FaultConfig = FaultConfig {
    mode: FaultMode::Normal,
    read_pin: DHT_PIN,
    bypass_checksum: false,
    baseline_sigma: BASELINE_SIGMA,
    spike_magnitude_sigma_range: (0.8, 2.0),
    spike_magnitude_range: (1.0, 4.0),
    drift_rate_sigma_range: (0.0125, 0.05),
    drift_rate_range: (0.03125, 0.09375),
    event_duration_samples: 0,
};

pub const FAULT_SPIKE: FaultConfig = FaultConfig {
    mode: FaultMode::Spike,
    event_duration_samples: 2,
    ..FAULT_NORMAL
};

pub const FAULT_DRIFT: FaultConfig = FaultConfig {
    mode: FaultMode::Drift,
    event_duration_samples: 20,
    ..FAULT_NORMAL
};

pub const FAULT_STUCK: FaultConfig = FaultConfig {
    mode: FaultMode::Stuck,
    event_duration_samples: 10,
    ..FAULT_NORMAL
};

pub const FAULT_ALL: FaultConfig = FaultConfig {
    mode: FaultMode::All,
    ..FAULT_NORMAL
};

#[derive(Clone, Copy, Debug)]
pub struct Neighbor {
    pub device_id: &'static str,
    pub node_index: usize,
    pub mac: [u8; 6],
}
