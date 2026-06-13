#![allow(dead_code)]

use embedded_svc::mqtt::client::EventPayload;
use esp_idf_svc::mqtt::client::{EspMqttClient, MqttClientConfiguration};
use log::info;

use crate::config;

pub fn connect() -> EspMqttClient<'static> {
    let broker_url = if config::MQTT_USER.is_empty() {
        format!("mqtt://{}:{}", config::MQTT_SERVER, config::MQTT_PORT)
    } else {
        format!(
            "mqtt://{}:{}@{}:{}",
            config::MQTT_USER, config::MQTT_PASSWORD, config::MQTT_SERVER, config::MQTT_PORT
        )
    };

    let mqtt_config = MqttClientConfiguration {
        client_id: Some(config::DEVICE_ID),
        ..Default::default()
    };

    info!("[MQTT] Connecting to {}:{}", config::MQTT_SERVER, config::MQTT_PORT);

    let client = EspMqttClient::new_cb(
        &broker_url,
        &mqtt_config,
        |event| match event.payload() {
            EventPayload::Connected(session_present) => {
                info!("[MQTT] connected session_present={}", session_present);
            }
            EventPayload::Published(message_id) => {
                info!("[MQTT] publish acknowledged message_id={}", message_id);
            }
            EventPayload::Error(error) => {
                log::error!("[MQTT] event error: {:?}", error);
            }
            payload => {
                log::debug!("[MQTT] Event: {:?}", payload);
            }
        },
    )
    .expect("Failed to create MQTT client");

    info!("[MQTT] client created");
    client
}
