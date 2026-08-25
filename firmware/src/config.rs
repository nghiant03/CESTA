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
