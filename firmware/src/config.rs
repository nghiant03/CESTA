#![allow(dead_code)]

use crate::fault::{FaultConfig, FaultMode};

pub const WIFI_SSID: &str = "YOUR_WIFI_NAME";
pub const WIFI_PASSWORD: &str = "YOUR_WIFI_PASSWORD";

pub const MQTT_SERVER: &str = "YOUR_SERVER_IP";
pub const MQTT_PORT: u16 = 1883;
pub const MQTT_USER: &str = "";
pub const MQTT_PASSWORD: &str = "";

pub const DEVICE_ID: &str = "esp32_X";
pub const MQTT_TOPIC_PREFIX: &str = "cesta/readings/";

/// Graph node index of this device; must match the `--receiver-index` of the
/// exported node model in `firmware/model/model.json`.
pub const NODE_INDEX: usize = 0;

/// Graph senders of this device in `sender_indices` order: the devices this node
/// may request hidden-state payloads from. Each `device_id` must match the
/// neighbor's `DEVICE_ID`, each `node_index` must match the exported model's
/// `sender_indices`, and each `mac` must be the neighbor's station MAC (logged
/// by every node at boot as "[ESPNOW] own MAC ...").
pub const NEIGHBORS: [Neighbor; 2] = [
    Neighbor {
        device_id: "esp32_Y",
        node_index: 1,
        mac: [0x00; 6],
    },
    Neighbor {
        device_id: "esp32_Z",
        node_index: 2,
        mac: [0x00; 6],
    },
];

/// Time the diagnosis cycle waits for all requested neighbor payloads.
pub const EXCHANGE_WAIT_MS: u64 = 1500;

/// Poll interval of the ESP-NOW and MQTT worker loops and of the response
/// collection loop.
pub const EXCHANGE_POLL_MS: u64 = 50;

/// MQTT in/out buffer size for telemetry JSON.
pub const MQTT_BUFFER_BYTES: usize = 8 * 1024;

pub const DHT_PIN: i32 = 5;
pub const SPIKE_DHT_PIN: i32 = 7;
pub const SEND_INTERVAL_SECS: u64 = 3;

pub const INFERENCE_ENABLED: bool = true;
pub const INFERENCE_TENSOR_ARENA_BYTES: usize = 2 * 1024 * 1024;

pub const NTP_SERVER: &str = "vn.pool.ntp.org";
pub const NTP_SYNC_TIMEOUT_SECS: u64 = 5;
pub const NTP_SYNC_POLL_MS: u64 = 500;

pub const FAULT_CONFIG: FaultConfig = FAULT_NORMAL;

pub const FAULT_NORMAL: FaultConfig = FaultConfig {
    mode: FaultMode::Normal,
    read_pin: DHT_PIN,
    bypass_checksum: false,
};

pub const FAULT_SPIKE: FaultConfig = FaultConfig {
    mode: FaultMode::Spike,
    read_pin: SPIKE_DHT_PIN,
    bypass_checksum: true,
};

#[derive(Clone, Copy, Debug)]
pub struct Neighbor {
    pub device_id: &'static str,
    pub node_index: usize,
    /// Station MAC address of the neighbor, used for direct ESP-NOW exchange.
    pub mac: [u8; 6],
}
