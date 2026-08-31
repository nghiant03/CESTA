mod config;
mod dht;
mod exchange;
mod fault;
mod inference;
mod mqtt;
mod wifi;

use std::sync::mpsc::Sender;
use std::thread;
use std::time::{Duration, Instant};

use esp_idf_svc::sntp::{EspSntp, OperatingMode, SntpConf, SyncMode, SyncStatus};
use log::info;

use crate::inference::NodeClassifier;
use crate::mqtt::PublishJob;

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

    let mut classifier = if config::INFERENCE_ENABLED {
        let sender_indices: Vec<usize> = config::NEIGHBORS.iter().map(|neighbor| neighbor.node_index).collect();
        match inference::NodeClassifier::new(config::INFERENCE_TENSOR_ARENA_BYTES, config::NODE_INDEX, &sender_indices)
        {
            Ok(classifier) => {
                info!(
                    "[INFERENCE] node model initialized receiver_index={} neighbors={} hidden_size={} mode={}",
                    classifier.receiver_index(),
                    classifier.neighbor_count(),
                    classifier.hidden_size(),
                    classifier.communication_mode()
                );
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

    if let Some(classifier) = classifier.as_ref() {
        exchange::init(
            classifier.window_size(),
            classifier.hidden_size(),
            classifier.features_per_node(),
        );
    }
    let publisher = mqtt::start();

    let topic = format!("{}{}", config::MQTT_TOPIC_PREFIX, config::DEVICE_ID);

    loop {
        match normal_dht_sensor.read(true) {
            Ok(reading) => {
                if let Some(classifier) = classifier.as_mut()
                    && classifier.push_temperature(reading.temperature)
                {
                    run_diagnosis_cycle(classifier, &publisher, &topic);
                }
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
                publish_json(&publisher, &topic, payload);
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
                    publish_json(&publisher, &topic, payload);
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

/// One distributed CESTA diagnosis cycle: receiver-local request pass, neighbor
/// exchange over MQTT, aggregate pass with received payloads, and a diagnosis
/// telemetry message.
fn run_diagnosis_cycle(classifier: &mut NodeClassifier, publisher: &Sender<PublishJob>, topic: &str) {
    let request_pass = match classifier.predict(None) {
        Ok(pass) => pass,
        Err(error) => {
            log::error!("[INFERENCE] request pass failed: {}", error);
            return;
        }
    };
    let window_id = exchange::update_cache(&request_pass.hidden, classifier.features());
    let requested = classifier.threshold_requests(&request_pass.request);

    let mut outstanding = 0;
    for (neighbor, timesteps) in requested.iter().enumerate() {
        outstanding += timesteps.len();
        if timesteps.is_empty() {
            continue;
        }
        let payload = exchange::encode_request(config::DEVICE_ID, window_id, timesteps);
        let request_topic = exchange::request_topic(config::NEIGHBORS[neighbor].device_id);
        if publisher
            .send(PublishJob {
                topic: request_topic,
                payload,
            })
            .is_err()
        {
            log::error!(
                "[EXCHANGE] failed to queue request to {}",
                config::NEIGHBORS[neighbor].device_id
            );
        }
    }

    let mut slots = classifier.slots();
    let deadline = Instant::now() + Duration::from_millis(config::EXCHANGE_WAIT_MS);
    while outstanding > 0 && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(config::EXCHANGE_POLL_MS));
        for response in exchange::take_responses() {
            if response.window_id != window_id {
                continue;
            }
            let Some(neighbor) = config::NEIGHBORS
                .iter()
                .position(|candidate| candidate.device_id == response.responder)
            else {
                log::warn!("[EXCHANGE] response from unknown device {}", response.responder);
                continue;
            };
            for (index, timestep) in response.timesteps.iter().enumerate() {
                if !requested[neighbor].contains(timestep) {
                    continue;
                }
                let timestep = *timestep as usize;
                let hidden_start = index * classifier.hidden_size();
                let features_start = index * classifier.features_per_node();
                let hidden = &response.hidden[hidden_start..hidden_start + classifier.hidden_size()];
                let features = &response.features[features_start..features_start + classifier.features_per_node()];
                if slots.fill(timestep, neighbor, hidden, features) {
                    outstanding = outstanding.saturating_sub(1);
                }
            }
        }
    }

    let aggregate_pass = match classifier.predict(Some(&slots)) {
        Ok(pass) => pass,
        Err(error) => {
            log::error!("[INFERENCE] aggregate pass failed: {}", error);
            return;
        }
    };
    let diagnosis = classifier.diagnosis(&aggregate_pass);
    let requested_summary: Vec<(&str, usize)> = requested
        .iter()
        .enumerate()
        .map(|(neighbor, timesteps)| (config::NEIGHBORS[neighbor].device_id, timesteps.len()))
        .collect();
    let received_summary: Vec<(&str, usize)> = (0..classifier.neighbor_count())
        .map(|neighbor| {
            (
                config::NEIGHBORS[neighbor].device_id,
                (0..classifier.window_size())
                    .filter(|timestep| slots.is_received(*timestep, neighbor))
                    .count(),
            )
        })
        .collect();

    info!(
        "[INFERENCE] window_id={} class={} label={} confidence={:.4} requested_timesteps={} received_timesteps={} request_ms={} aggregate_ms={}",
        window_id,
        diagnosis.class,
        diagnosis.label,
        diagnosis.confidence,
        requested.iter().map(Vec::len).sum::<usize>(),
        slots.received_count(),
        request_pass.elapsed_ms,
        aggregate_pass.elapsed_ms
    );
    let payload = serde_json::json!({
        "device_id": config::DEVICE_ID,
        "timestamp": timestamp_epoch(),
        "type": "inference",
        "window_id": window_id,
        "communication_mode": classifier.communication_mode(),
        "label": diagnosis.label,
        "class": diagnosis.class,
        "confidence": diagnosis.confidence,
        "probabilities": diagnosis.probabilities,
        "requested": requested_summary,
        "received": received_summary,
        "request_elapsed_ms": request_pass.elapsed_ms,
        "aggregate_elapsed_ms": aggregate_pass.elapsed_ms,
    });
    publish_json(publisher, topic, payload);
}

fn publish_json(publisher: &Sender<PublishJob>, topic: &str, payload: serde_json::Value) {
    let message = serde_json::to_string(&payload).unwrap();
    if publisher
        .send(PublishJob {
            topic: topic.to_owned(),
            payload: message.clone().into_bytes(),
        })
        .is_err()
    {
        log::error!("[MQTT] publish queue closed topic={}", topic);
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
