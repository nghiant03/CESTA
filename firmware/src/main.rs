mod config;
mod dht;
mod fault;
mod inference;
mod mqtt;
mod wifi;

use std::thread;
use std::time::Duration;

use esp_idf_svc::sntp::{EspSntp, OperatingMode, SntpConf, SyncMode, SyncStatus};
use log::info;

fn main() {
    esp_idf_svc::sys::link_patches();
    esp_idf_svc::log::EspLogger::initialize_default();

    info!(
        "[CESTA] Device {} starting with fault mode {} ",
        config::DEVICE_ID,
        config::FAULT_CONFIG.mode.as_str(),
    );

    let _peripherals = esp_idf_hal::peripherals::Peripherals::take().unwrap();
    let dht_pin = unsafe { esp_idf_hal::gpio::AnyIOPin::new(config::DHT_PIN) };
    let mut normal_dht_sensor = dht::Dht11Sensor::new(dht_pin);
    let mut fault_dht_sensor = if config::FAULT_CONFIG.uses_fault_read_pin(config::DHT_PIN) {
        Some(dht::Dht11Sensor::new(unsafe {
            esp_idf_hal::gpio::AnyIOPin::new(config::FAULT_CONFIG.read_pin)
        }))
    } else {
        None
    };

    let _wifi = wifi::connect();
    sync_time();

    let topic = format!("{}{}", config::MQTT_TOPIC_PREFIX, config::DEVICE_ID);
    let mut mqtt_client = mqtt::connect();
    let mut classifier = if config::INFERENCE_ENABLED {
        match inference::Classifier::new(config::INFERENCE_TENSOR_ARENA_BYTES) {
            Ok(classifier) => {
                info!("[INFERENCE] TensorFlow Lite model initialized");
                Some(classifier)
            }
            Err(error) => {
                log::error!("[INFERENCE] Initialization failed: {}", error);
                None
            }
        }
    } else {
        None
    };

    loop {
        match normal_dht_sensor.read(true) {
            Ok(reading) => {
                let payload = serde_json::json!({
                    "device_id": config::DEVICE_ID,
                    "timestamp": timestamp_epoch(),
                    "temperature": reading.temperature,
                    "humidity": reading.humidity,
                    "path": "normal",
                    "fault_mode": "normal",
                    "gpio": config::DHT_PIN,
                });
                info!(
                    "[DHT] path=normal gpio={} temperature={:.1}°C humidity={:.1}% payload={}",
                    config::DHT_PIN,
                    reading.temperature,
                    reading.humidity,
                    payload
                );
                if let Some(classifier) = classifier.as_mut()
                    && let Some(result) = classifier.push_temperature(reading.temperature)
                {
                    match result {
                        Ok(result) => info!(
                            "[INFERENCE] class={} confidence={:.4} elapsed_ms={} scores={:?}",
                            result.class, result.confidence, result.elapsed_ms, result.scores
                        ),
                        Err(error) => log::error!("[INFERENCE] Prediction failed: {}", error),
                    }
                }
                let msg = serde_json::to_string(&payload).unwrap();

                match mqtt_client.publish(
                    &topic,
                    esp_idf_svc::mqtt::client::QoS::AtLeastOnce,
                    false,
                    msg.as_bytes(),
                ) {
                    Ok(message_id) => info!(
                        "[MQTT] publish queued message_id={} topic={} payload={}",
                        message_id, topic, msg
                    ),
                    Err(e) => log::error!("[MQTT] Publish failed: {:?}", e),
                }
            }
            Err(e) => {
                log::warn!(
                    "[DHT] path=normal gpio={} read failed: {:?}, skipping",
                    config::DHT_PIN,
                    e
                );
            }
        }

        if let Some(sensor) = fault_dht_sensor.as_mut() {
            match sensor.read(config::FAULT_CONFIG.checksum_enabled()) {
                Ok(reading) => {
                    let payload = serde_json::json!({
                        "device_id": config::DEVICE_ID,
                        "timestamp": timestamp_epoch(),
                        "temperature": reading.temperature,
                        "humidity": reading.humidity,
                        "path": "fault",
                        "fault_mode": config::FAULT_CONFIG.mode.as_str(),
                        "gpio": config::FAULT_CONFIG.read_pin,
                    });
                    info!(
                        "[DHT] path=fault fault={} gpio={} temperature={:.1}°C humidity={:.1}% payload={}",
                        config::FAULT_CONFIG.mode.as_str(),
                        config::FAULT_CONFIG.read_pin,
                        reading.temperature,
                        reading.humidity,
                        payload
                    );
                    let msg = serde_json::to_string(&payload).unwrap();

                    match mqtt_client.publish(
                        &topic,
                        esp_idf_svc::mqtt::client::QoS::AtLeastOnce,
                        false,
                        msg.as_bytes(),
                    ) {
                        Ok(message_id) => info!(
                            "[MQTT] publish queued message_id={} topic={} payload={}",
                            message_id, topic, msg
                        ),
                        Err(e) => log::error!("[MQTT] Publish failed: {:?}", e),
                    }
                }
                Err(e) => {
                    log::warn!(
                        "[DHT] path=fault fault={} gpio={} read failed: {:?}, skipping",
                        config::FAULT_CONFIG.mode.as_str(),
                        config::FAULT_CONFIG.read_pin,
                        e
                    );
                }
            }
        }

        thread::sleep(Duration::from_secs(config::SEND_INTERVAL_SECS));
    }
}

fn sync_time() {
    let timeout_ms = config::NTP_SYNC_TIMEOUT_SECS * 1_000;
    let poll_ms = config::NTP_SYNC_POLL_MS;
    let max_attempts = timeout_ms.div_ceil(poll_ms);

    info!(
        "[NTP] Syncing time from {} with timeout={} ms poll={} ms max_attempts={}",
        config::NTP_SERVER,
        timeout_ms,
        poll_ms,
        max_attempts
    );
    let conf = SntpConf {
        servers: [config::NTP_SERVER],
        operating_mode: OperatingMode::Poll,
        sync_mode: SyncMode::Immediate,
    };
    let sntp = EspSntp::new(&conf).expect("Failed to create SNTP client");

    let mut attempts = 0;
    let mut elapsed_ms = 0;
    while sntp.get_sync_status() != SyncStatus::Completed && attempts < max_attempts {
        thread::sleep(Duration::from_millis(poll_ms));
        attempts += 1;
        elapsed_ms += poll_ms;
    }

    if sntp.get_sync_status() == SyncStatus::Completed {
        info!("[NTP] Time synced after {} ms", elapsed_ms);
    } else {
        log::warn!(
            "[NTP] Sync timed out after {} ms using {}, final status {:?}, timestamps may use uptime",
            elapsed_ms,
            config::NTP_SERVER,
            sntp.get_sync_status()
        );
    }

    std::mem::forget(sntp);
}

fn timestamp_epoch() -> u64 {
    use std::time::SystemTime;
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
