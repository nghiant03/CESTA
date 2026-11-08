//! MQTT worker.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, TrySendError, sync_channel};
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

static QUEUE_DROPS: AtomicU32 = AtomicU32::new(0);
static PUBLISH_FAILURES: AtomicU32 = AtomicU32::new(0);
static ACKNOWLEDGED: AtomicU32 = AtomicU32::new(0);

pub struct Publisher(SyncSender<PublishJob>);

impl Publisher {
    pub fn send(&self, job: PublishJob) -> Result<(), TrySendError<PublishJob>> {
        self.0.try_send(job).inspect_err(|_| {
            QUEUE_DROPS.fetch_add(1, Ordering::Relaxed);
        })
    }
}

pub fn telemetry_snapshot() -> serde_json::Value {
    serde_json::json!({
        "queue_drops": QUEUE_DROPS.load(Ordering::Relaxed),
        "publish_failures": PUBLISH_FAILURES.load(Ordering::Relaxed),
        "acknowledged": ACKNOWLEDGED.load(Ordering::Relaxed),
    })
}

pub fn start() -> Publisher {
    let (sender, receiver) = sync_channel(config::MQTT_QUEUE_CAPACITY);
    thread::Builder::new()
        .stack_size(16 * 1024)
        .spawn(move || run(receiver))
        .expect("Failed to spawn MQTT worker");
    Publisher(sender)
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
            ACKNOWLEDGED.fetch_add(1, Ordering::Relaxed);
            info!("[MQTT] publish acknowledged message_id={}", message_id);
        }
        EventPayload::Error(_) => {
            error!("[MQTT] event error");
        }
        _ => {}
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
        Err(error) => {
            PUBLISH_FAILURES.fetch_add(1, Ordering::Relaxed);
            error!("[MQTT] publish failed topic={}: {:?}", topic, error);
        }
    }
}
