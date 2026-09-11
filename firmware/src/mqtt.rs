//! MQTT worker: owns the client and publishes queued telemetry. The neighbor
//! exchange no longer uses MQTT; it runs over ESP-NOW (see `espnow.rs`).

use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread;
use std::time::Duration;

use embedded_svc::mqtt::client::EventPayload;
use esp_idf_svc::mqtt::client::{EspMqttClient, MqttClientConfiguration, QoS};
use log::{debug, error, info};

use crate::config;

pub struct PublishJob {
    pub topic: String,
    pub payload: Vec<u8>,
}

/// Start the MQTT worker and return the publish queue.
pub fn start() -> Sender<PublishJob> {
    let (sender, receiver) = channel();
    thread::Builder::new()
        .stack_size(16 * 1024)
        .spawn(move || run(receiver))
        .expect("Failed to spawn MQTT worker");
    sender
}

fn run(receiver: Receiver<PublishJob>) {
    let mut client = connect();
    loop {
        for job in receiver.try_iter() {
            publish(&mut client, &job.topic, &job.payload);
        }
        thread::sleep(Duration::from_millis(config::EXCHANGE_POLL_MS));
    }
}

fn connect() -> EspMqttClient<'static> {
    let broker_url = if config::MQTT_USER.is_empty() {
        format!("mqtt://{}:{}", config::MQTT_SERVER, config::MQTT_PORT)
    } else {
        format!(
            "mqtt://{}:{}@{}:{}",
            config::MQTT_USER,
            config::MQTT_PASSWORD,
            config::MQTT_SERVER,
            config::MQTT_PORT
        )
    };

    let mqtt_config = MqttClientConfiguration {
        client_id: Some(config::DEVICE_ID),
        buffer_size: config::MQTT_BUFFER_BYTES,
        out_buffer_size: config::MQTT_BUFFER_BYTES,
        ..Default::default()
    };

    info!(
        "[MQTT] Connecting to {}:{}",
        config::MQTT_SERVER,
        config::MQTT_PORT
    );

    let client = EspMqttClient::new_cb(&broker_url, &mqtt_config, |event| match event.payload() {
        EventPayload::Connected(session_present) => {
            info!("[MQTT] connected session_present={}", session_present);
        }
        EventPayload::Disconnected => {
            info!("[MQTT] disconnected");
        }
        EventPayload::Published(message_id) => {
            info!("[MQTT] publish acknowledged message_id={}", message_id);
        }
        EventPayload::Error(error) => {
            error!("[MQTT] event error: {:?}", error);
        }
        payload => {
            debug!("[MQTT] Event: {:?}", payload);
        }
    })
    .expect("Failed to create MQTT client");

    info!("[MQTT] client created");
    client
}

fn publish(client: &mut EspMqttClient<'static>, topic: &str, payload: &[u8]) {
    match client.publish(topic, QoS::AtLeastOnce, false, payload) {
        Ok(message_id) => {
            debug!(
                "[MQTT] published message_id={} topic={} bytes={}",
                message_id,
                topic,
                payload.len()
            );
        }
        Err(error) => error!("[MQTT] publish failed topic={}: {:?}", topic, error),
    }
}
