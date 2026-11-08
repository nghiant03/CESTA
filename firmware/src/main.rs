//! Node firmware.

mod config;
mod dht;
mod diagnostic;
mod espnow;
mod exchange;
mod fault;
mod inference;
mod mqtt;
mod telemetry;
mod wifi;

use std::collections::VecDeque;
use std::thread;
use std::time::{Duration, Instant};

use esp_idf_svc::sntp::{EspSntp, OperatingMode, SntpConf, SyncMode, SyncStatus};
use log::info;

use crate::inference::NodeClassifier;
use crate::mqtt::PublishJob;

fn main() {
    esp_idf_svc::sys::link_patches();
    esp_idf_svc::log::EspLogger::initialize_default();

    if config::INFERENCE_SYNTHETIC_DIAGNOSTIC {
        diagnostic::run();
    }

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
    let mut fault_injector = if config::FAULT_CONFIG.mode.is_software_injected() {
        Some(fault::FaultInjector::new(config::FAULT_CONFIG))
    } else {
        None
    };

    let _wifi = wifi::connect();
    let _sntp = sync_time();

    let mut classifier = if config::INFERENCE_ENABLED {
        let sender_indices: Vec<usize> = config::NEIGHBORS
            .iter()
            .map(|neighbor| neighbor.node_index)
            .collect();
        match inference::NodeClassifier::new(
            config::INFERENCE_TENSOR_ARENA_BYTES,
            config::NODE_INDEX,
            &sender_indices,
        ) {
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

    if let Some(active) = classifier.as_ref() {
        if let Err(error) = exchange::init(
            active.window_size(),
            active.hidden_size(),
            active.features_per_node(),
            active.model_fingerprint(),
        ) {
            log::error!(
                "[INFERENCE] exchange compatibility validation failed: {}",
                error
            );
            classifier = None;
        }
    }
    let espnow = espnow::start();
    let publisher = mqtt::start();
    let mut telemetry = telemetry::Telemetry::new();
    telemetry.status(
        &publisher,
        classifier.as_ref(),
        timestamp_epoch(),
        "startup",
    );
    if classifier.is_none() {
        telemetry.status(&publisher, None, timestamp_epoch(), "inference_unavailable");
    }

    let topic = format!("{}{}", config::MQTT_TOPIC_PREFIX, config::DEVICE_ID);

    let cadence = if let Some(classifier) = classifier.as_ref() {
        classifier.sample_interval_secs()
    } else if config::INFERENCE_ENABLED || fault_injector.is_some() {
        60
    } else {
        config::SEND_INTERVAL_SECS
    };
    let sample_interval = Duration::from_secs(cadence);
    let mut next_slot = (timestamp_epoch() / cadence + 1) * cadence;
    let mut previous_slot = None;
    let mut emitted_labels = VecDeque::new();
    info!(
        "[DHT] cadence_seconds={} inference_enabled={} aligned_epoch_slots=true",
        cadence,
        config::INFERENCE_ENABLED
    );
    loop {
        let sample_timestamp = wait_for_slot(&mut next_slot, cadence);
        telemetry.samples_observed += 1;
        if previous_slot.is_some_and(|previous| sample_timestamp != previous + cadence) {
            log::warn!("[INFERENCE] skipped sample slot; dropping incomplete window");
            telemetry.sample_resets += 1;
            telemetry.status(
                &publisher,
                classifier.as_ref(),
                sample_timestamp,
                "sample_slot_missed",
            );
            reset_window(&mut classifier, &mut emitted_labels);
        }
        previous_slot = Some(sample_timestamp);
        let sample_started = Instant::now();
        match normal_dht_sensor.read(true) {
            Ok(reading) => {
                let injected = fault_injector.as_mut().map(|injector| {
                    let temperature = injector.apply(reading.temperature);
                    (temperature, injector.emitted_mode())
                });
                let inference_temperature = injected
                    .map(|(temperature, _)| temperature)
                    .unwrap_or(reading.temperature);
                let read_finished = timestamp_epoch();
                let sample_valid = read_finished >= sample_timestamp
                    && read_finished < sample_timestamp + cadence
                    && sample_started.elapsed() < sample_interval;
                if !sample_valid {
                    telemetry.sample_resets += 1;
                    telemetry.status(
                        &publisher,
                        classifier.as_ref(),
                        sample_timestamp,
                        "sample_late",
                    );
                    reset_window(&mut classifier, &mut emitted_labels);
                } else if let Some(classifier) = classifier.as_mut() {
                    if emitted_labels.len() == classifier.window_size() {
                        emitted_labels.pop_front();
                    }
                    emitted_labels
                        .push_back(injected.map(|(_, mode)| mode.as_str()).unwrap_or("normal"));
                    if classifier.push_temperature(inference_temperature) {
                        run_diagnosis_cycle(
                            classifier,
                            &espnow,
                            &publisher,
                            &topic,
                            sample_timestamp,
                            &emitted_labels,
                            &mut telemetry,
                        );
                    } else {
                        telemetry.status(
                            &publisher,
                            Some(classifier),
                            sample_timestamp,
                            "window_warmup",
                        );
                    }
                } else {
                    telemetry.status(&publisher, None, sample_timestamp, "inference_unavailable");
                }
                let payload = serde_json::json!({
                    "device_id": config::DEVICE_ID,
                    "timestamp": sample_timestamp,
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

                if let Some((faulted_temperature, fault_mode)) = injected {
                    let payload = serde_json::json!({
                        "device_id": config::DEVICE_ID,
                        "timestamp": sample_timestamp,
                        "temperature": faulted_temperature,
                        "humidity": reading.humidity,
                        "path": "fault",
                        "fault_mode": fault_mode.as_str(),
                        "gpio": config::FAULT_CONFIG.read_pin,
                        "source": "software",
                    });
                    info!(
                        "[DHT] path=fault fault={} source=software temperature={:.1}°C (base {:.1}°C) payload={}",
                        fault_mode.as_str(),
                        faulted_temperature,
                        reading.temperature,
                        payload
                    );
                    publish_json(&publisher, &topic, payload);
                }
            }
            Err(e) => {
                telemetry.sample_resets += 1;
                telemetry.status(
                    &publisher,
                    classifier.as_ref(),
                    sample_timestamp,
                    "sample_read_failed",
                );
                reset_window(&mut classifier, &mut emitted_labels);
                log::warn!(
                    "[DHT] path=normal gpio={} read failed: {:?}, skipping",
                    config::DHT_PIN,
                    e
                );
            }
        }

        if let Some(sensor) = fault_dht_sensor.as_mut() {
            let sample_timestamp = timestamp_epoch();
            match sensor.read(config::FAULT_CONFIG.checksum_enabled()) {
                Ok(reading) => {
                    let payload = serde_json::json!({
                        "device_id": config::DEVICE_ID,
                        "timestamp": sample_timestamp,
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
    }
}

fn reset_window(classifier: &mut Option<NodeClassifier>, labels: &mut VecDeque<&str>) {
    if let Some(classifier) = classifier.as_mut() {
        classifier.reset_window();
    }
    labels.clear();
    exchange::invalidate_cache();
}

fn wait_for_slot(next_slot: &mut u64, cadence: u64) -> u64 {
    loop {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::SystemTime::UNIX_EPOCH)
            .unwrap_or_default();
        let target = Duration::from_secs(*next_slot);
        if now < target && target - now > Duration::from_secs(cadence) {
            *next_slot = (now.as_secs() / cadence + 1) * cadence;
        } else if now < target {
            thread::sleep((target - now).min(Duration::from_millis(200)));
        } else if now - target > Duration::from_secs(2) {
            *next_slot = (now.as_secs() / cadence + 1) * cadence;
        } else {
            let slot = *next_slot;
            *next_slot += cadence;
            return slot;
        }
    }
}

/// One distributed CESTA diagnosis cycle: receiver-local request pass, neighbor
/// exchange over ESP-NOW, aggregate pass with received payloads, and a
/// diagnosis telemetry message.
fn run_diagnosis_cycle(
    classifier: &mut NodeClassifier,
    espnow: &espnow::EspNowSender,
    publisher: &mqtt::Publisher,
    topic: &str,
    window_id: u64,
    emitted_labels: &VecDeque<&str>,
    telemetry: &mut telemetry::Telemetry,
) {
    let window_ready_epoch_us = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64;
    let cycle_started = Instant::now();
    telemetry.cycles_attempted += 1;
    let available = espnow::registered_peer_mask();
    for (neighbor, available) in available.iter().enumerate() {
        if let Err(error) = classifier.set_neighbor_available(neighbor, *available) {
            log::error!("[INFERENCE] peer availability invalid: {}", error);
            telemetry.emit(
                publisher,
                serde_json::json!({
                    "type": "benchmark_status",
                    "timestamp": window_id,
                    "checkpoint_sha256": classifier.model_fingerprint(),
                    "reason": "peer_availability_invalid",
                    "failed_stage": "peer_availability",
                    "timing": {
                        "cycle_us": cycle_started.elapsed().as_micros() as u64,
                        "request_inference_us": Option::<u64>::None,
                        "exchange_wait_us": Option::<u64>::None,
                        "aggregate_inference_us": Option::<u64>::None,
                    },
                }),
            );
            return;
        }
    }
    let request_started = Instant::now();
    let request_pass =
        match classifier.predict(None) {
            Ok(pass) => pass,
            Err(error) => {
                log::error!("[INFERENCE] request pass failed: {}", error);
                telemetry.emit(publisher, serde_json::json!({
                "type": "benchmark_status",
                "timestamp": window_id,
                "checkpoint_sha256": classifier.model_fingerprint(),
                "reason": "request_inference_failed",
                "failed_stage": "request_inference",
                "timing": {
                    "cycle_us": cycle_started.elapsed().as_micros() as u64,
                    "request_inference_us": Some(request_started.elapsed().as_micros() as u64),
                    "exchange_wait_us": Option::<u64>::None,
                    "aggregate_inference_us": Option::<u64>::None,
                },
            }));
                return;
            }
        };
    let request_inference_us = request_started.elapsed().as_micros() as u64;
    let exchange_started = Instant::now();
    if !exchange::update_cache(window_id, &request_pass.hidden, classifier.features()) {
        log::error!("[EXCHANGE] refusing invalid window cache");
        telemetry.emit(
            publisher,
            serde_json::json!({
                "type": "benchmark_status",
                "timestamp": window_id,
                "checkpoint_sha256": classifier.model_fingerprint(),
                "reason": "window_cache_invalid",
                "failed_stage": "exchange",
                "timing": {
                    "cycle_us": cycle_started.elapsed().as_micros() as u64,
                    "request_inference_us": Some(request_inference_us),
                    "exchange_wait_us": Some(exchange_started.elapsed().as_micros() as u64),
                    "aggregate_inference_us": Option::<u64>::None,
                },
            }),
        );
        return;
    }
    let requested = classifier.threshold_requests(&request_pass.request);

    let request_nonce = unsafe {
        ((esp_idf_svc::sys::esp_random() as u64) << 32) | esp_idf_svc::sys::esp_random() as u64
    };
    exchange::begin_response_cycle(window_id, request_nonce);
    let mut outstanding = 0;
    for (neighbor, timesteps) in requested.iter().enumerate() {
        outstanding += timesteps.len();
        if timesteps.is_empty() {
            continue;
        }
        let payload =
            exchange::encode_request(config::DEVICE_ID, window_id, request_nonce, timesteps);
        if espnow
            .send(espnow::EspNowJob {
                target: config::NEIGHBORS[neighbor].mac,
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
            if response.window_id != window_id || response.request_nonce != request_nonce {
                continue;
            }
            let Some(neighbor) = config::NEIGHBORS
                .iter()
                .position(|candidate| candidate.device_id == response.responder)
            else {
                log::warn!(
                    "[EXCHANGE] response from unknown device {}",
                    response.responder
                );
                continue;
            };
            for (index, timestep) in response.timesteps.iter().enumerate() {
                if !requested[neighbor].contains(timestep) {
                    continue;
                }
                let timestep = *timestep as usize;
                let hidden_start = index * classifier.hidden_size();
                let features_start = index * classifier.features_per_node();
                let hidden =
                    &response.hidden[hidden_start..hidden_start + classifier.hidden_size()];
                let features = &response.features
                    [features_start..features_start + classifier.features_per_node()];
                if slots.fill(timestep, neighbor, hidden, features) {
                    outstanding = outstanding.saturating_sub(1);
                }
            }
        }
    }

    exchange::end_response_cycle();
    let exchange_wait_us = exchange_started.elapsed().as_micros() as u64;
    let aggregate_started = Instant::now();
    let aggregate_pass = match classifier.predict(Some(&slots)) {
        Ok(pass) => pass,
        Err(error) => {
            log::error!("[INFERENCE] aggregate pass failed: {}", error);
            telemetry.emit(publisher, serde_json::json!({
                "type": "benchmark_status",
                "timestamp": window_id,
                "checkpoint_sha256": classifier.model_fingerprint(),
                "reason": "aggregate_inference_failed",
                "failed_stage": "aggregate_inference",
                "timing": {
                    "cycle_us": cycle_started.elapsed().as_micros() as u64,
                    "request_inference_us": Some(request_inference_us),
                    "exchange_wait_us": Some(exchange_wait_us),
                    "aggregate_inference_us": Some(aggregate_started.elapsed().as_micros() as u64),
                },
            }));
            return;
        }
    };
    let aggregate_inference_us = aggregate_started.elapsed().as_micros() as u64;
    let cycle_us = cycle_started.elapsed().as_micros() as u64;
    let decision_ready_epoch_us = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64;
    telemetry.cycles_completed += 1;
    let payload_bytes_per_message =
        (classifier.hidden_size() + classifier.features_per_node()) * std::mem::size_of::<f32>();
    let edges: Vec<serde_json::Value> = config::NEIGHBORS.iter().enumerate().map(|(neighbor, peer)| {
        serde_json::json!({
            "sender_node_index": peer.node_index,
            "possible_messages": if available[neighbor] { classifier.window_size() } else { 0 },
            "requested_messages": requested[neighbor].len(),
            "received_messages": (0..classifier.window_size()).filter(|timestep| slots.is_received(*timestep, neighbor)).count(),
        })
    }).collect();
    telemetry.emit(publisher, serde_json::json!({
        "type": "benchmark",
        "decision_ready_epoch_us": decision_ready_epoch_us,
        "window_ready_epoch_us": window_ready_epoch_us,
        "window_size": classifier.window_size(),
        "checkpoint_sha256": classifier.model_fingerprint(),
        "window_start_timestamp": window_id - (classifier.window_size() as u64 - 1) * classifier.sample_interval_secs(),
        "window_end_timestamp": window_id,
        "sample_interval_seconds": classifier.sample_interval_secs(),
        "prediction_policy": "window_all",
        "timing": {
            "request_inference_us": request_inference_us,
            "exchange_wait_us": exchange_wait_us,
            "aggregate_inference_us": aggregate_inference_us,
            "cycle_us": cycle_us,
        },
        "memory": telemetry::memory(),
        "communication": {
            "purpose": "radio_energy_model_inputs",
            "edges": edges,
            "payload_bytes_per_message": payload_bytes_per_message,
        },
    }));
    let diagnosis = classifier.diagnosis(&aggregate_pass);
    let decoded_labels: Vec<&str> = aggregate_pass
        .decoded
        .iter()
        .map(|class| ["NORMAL", "SPIKE", "DRIFT", "STUCK"][*class])
        .collect();
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
        "window_start_timestamp": window_id - (classifier.window_size() as u64 - 1) * 60,
        "window_end_timestamp": window_id,
        "sample_interval_seconds": 60,
        "emitted_labels": emitted_labels,
        "decoded_classes": aggregate_pass.decoded,
        "decoded_labels": decoded_labels,
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

fn publish_json(publisher: &mqtt::Publisher, topic: &str, payload: serde_json::Value) {
    let message = serde_json::to_string(&payload).unwrap();
    if publisher
        .send(PublishJob {
            topic: topic.to_owned(),
            payload: message.into_bytes(),
        })
        .is_err()
    {
        log::error!("[MQTT] publish queue full or closed topic={}", topic);
    }
}

/// Block until SNTP time sync completes or the configured timeout elapses.
fn sync_time() -> EspSntp<'static> {
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

    let started = Instant::now();
    let mut attempts = 0;
    loop {
        let status = sntp.get_sync_status();
        if status == SyncStatus::Completed {
            info!(
                "[NTP] Time synced server={} after {} ms epoch={}",
                config::NTP_SERVER,
                started.elapsed().as_millis(),
                timestamp_epoch()
            );
            break;
        }
        if attempts >= max_attempts {
            log::warn!(
                "[NTP] Still waiting for {} after {} ms, status {:?}",
                config::NTP_SERVER,
                started.elapsed().as_millis(),
                status
            );
            if !config::INFERENCE_ENABLED {
                break;
            }
            attempts = 0;
        }
        thread::sleep(Duration::from_millis(poll_ms));
        attempts += 1;
    }
    sntp
}

/// Current epoch seconds; 0 while the clock is unsynchronized.
fn timestamp_epoch() -> u64 {
    use std::time::SystemTime;
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
