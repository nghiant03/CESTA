use serde_json::{Value, json};

use crate::{config, inference::NodeClassifier, mqtt};

pub struct Telemetry {
    boot_id: String,
    event_seq: u64,
    pub samples_observed: u64,
    pub sample_resets: u64,
    pub cycles_attempted: u64,
    pub cycles_completed: u64,
}

impl Telemetry {
    pub fn new() -> Self {
        let random = unsafe {
            ((esp_idf_svc::sys::esp_random() as u64) << 32) | esp_idf_svc::sys::esp_random() as u64
        };
        Self {
            boot_id: format!("{random:016x}"),
            event_seq: 0,
            samples_observed: 0,
            sample_resets: 0,
            cycles_attempted: 0,
            cycles_completed: 0,
        }
    }

    pub fn emit(&mut self, publisher: &mqtt::Publisher, mut payload: Value) {
        self.event_seq += 1;
        payload["schema_version"] = json!(1);
        payload["benchmark_id"] = json!(config::BENCHMARK_ID);
        payload["stream_id"] = json!(config::BENCHMARK_STREAM_ID);
        payload["device_id"] = json!(config::DEVICE_ID);
        payload["node_index"] = json!(config::NODE_INDEX);
        payload["boot_id"] = json!(self.boot_id);
        payload["event_seq"] = json!(self.event_seq);
        payload["model_name"] = json!(if cfg!(hifinet) { "hifinet" } else { "cesta" });
        payload["execution_location"] = json!("device");
        let topic = format!(
            "cesta/benchmarks/{}/{}",
            config::BENCHMARK_ID,
            config::DEVICE_ID
        );
        if publisher
            .send(mqtt::PublishJob {
                topic,
                payload: payload.to_string().into_bytes(),
            })
            .is_err()
        {
            log::warn!("[BENCHMARK] event dropped event_seq={}", self.event_seq);
        }
    }

    pub fn status(
        &mut self,
        publisher: &mqtt::Publisher,
        classifier: Option<&NodeClassifier>,
        timestamp: u64,
        reason: &str,
    ) {
        self.emit(
            publisher,
            json!({
                "type": "benchmark_status",
                "timestamp": timestamp,
                "checkpoint_sha256": classifier.map(|active| {
                    #[cfg(hifinet)]
                    { let _ = active; crate::inference::CHECKPOINT }
                    #[cfg(not(hifinet))]
                    { active.model_fingerprint() }
                }),
                "reason": reason,
            }),
        );
    }
}

pub fn memory() -> Value {
    use esp_idf_svc::sys::*;
    let internal = MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT;
    let psram = MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT;
    unsafe {
        json!({
            "free_internal_bytes": heap_caps_get_free_size(internal),
            "min_free_internal_bytes": heap_caps_get_minimum_free_size(internal),
            "largest_internal_block_bytes": heap_caps_get_largest_free_block(internal),
            "free_psram_bytes": heap_caps_get_free_size(psram),
            "min_free_psram_bytes": heap_caps_get_minimum_free_size(psram),
            "largest_psram_block_bytes": heap_caps_get_largest_free_block(psram),
            "tensor_arena_bytes": config::INFERENCE_TENSOR_ARENA_BYTES,
        })
    }
}
