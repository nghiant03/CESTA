use crate::{config, espnow, exchange, mqtt, telemetry, tflite::StageRunner};
use std::{
    collections::VecDeque,
    time::{Duration, Instant},
};
include!(concat!(env!("OUT_DIR"), "/hifinet_bundle.rs"));

pub struct NodeClassifier {
    runner: StageRunner,
    samples: Vec<f32>,
}

impl NodeClassifier {
    pub fn new(arena: usize, receiver: usize, senders: &[usize]) -> Result<Self, String> {
        if receiver != RECEIVER || senders != SENDERS {
            return Err(
                "HiFiNet bundle receiver/sender order disagrees with node configuration".into(),
            );
        }
        let mut runner = StageRunner::new(arena)?;
        for (bytes, input, output) in STAGES {
            runner.load(bytes, input, output)?;
        }
        runner.unload();
        Ok(Self {
            runner,
            samples: Vec::with_capacity(60),
        })
    }
    pub fn window_size(&self) -> usize {
        60
    }
    pub fn receiver_index(&self) -> usize {
        RECEIVER
    }
    pub fn hidden_size(&self) -> usize {
        WIDTHS[0]
    }
    pub fn features_per_node(&self) -> usize {
        1
    }
    pub fn neighbor_count(&self) -> usize {
        SENDERS.len()
    }
    pub fn communication_mode(&self) -> &str {
        "hifinet_three_round_dense"
    }
    pub fn sample_interval_secs(&self) -> u64 {
        60
    }
    pub fn model_fingerprint(&self) -> &str {
        FINGERPRINT
    }
    pub fn reset_window(&mut self) {
        self.samples.clear();
    }
    pub fn push_temperature(&mut self, value: f32) -> bool {
        if !value.is_finite() {
            self.reset_window();
            return false;
        }
        if self.samples.len() == 60 {
            self.samples.remove(0);
        }
        self.samples.push(value);
        self.samples.len() == 60
    }
    fn invoke(
        &mut self,
        stage: usize,
        input: &[f32],
        deadline: Instant,
    ) -> Result<Vec<f32>, String> {
        let (bytes, incoming, outgoing) = STAGES[stage];
        if Instant::now() >= deadline {
            return Err("cycle deadline before stage load".into());
        }
        self.runner.load(bytes, incoming, outgoing)?;
        let mut output = vec![0.0; 60 * outgoing];
        if Instant::now() >= deadline {
            self.runner.unload();
            return Err("cycle deadline before stage invocation".into());
        }
        self.runner.predict(input, &mut output)?;
        self.runner.unload();
        if Instant::now() >= deadline {
            return Err("cycle deadline after stage invocation".into());
        }
        Ok(output)
    }
    pub fn run_cycle(
        &mut self,
        radio: &espnow::EspNowSender,
        publisher: &mqtt::Publisher,
        topic: &str,
        window: u64,
        ready: u64,
        labels: &VecDeque<&str>,
        telemetry: &mut telemetry::Telemetry,
    ) {
        telemetry.cycles_attempted += 1;
        let started = Instant::now();
        let mut rounds = Vec::new();
        let mut stages = Vec::new();
        let mut failed_stage = "admission".to_owned();
        let result = (|| -> Result<serde_json::Value, String> {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|e| e.to_string())?;
            let close = Duration::from_secs(window.checked_add(60).ok_or("invalid window")?);
            let age = now.checked_sub(close).ok_or("clock before bucket close")?;
            let deadline =
                started + Duration::from_millis(config::EXCHANGE_CUTOFF_MS).saturating_sub(age);
            let cycle_deadline =
                started + Duration::from_millis(config::CYCLE_BUDGET_MS).saturating_sub(age);
            if started >= deadline {
                return Err("exchange admission deadline missed".into());
            }
            exchange::prepare_window(window, deadline);
            failed_stage = "temporal".into();
            let timer = Instant::now();
            let h0 = self.invoke(0, &self.samples.clone(), cycle_deadline)?;
            stages.push(serde_json::json!({"stage":"temporal", "elapsed_us":timer.elapsed().as_micros() as u64}));
            let mut hidden = h0.clone();
            let available = espnow::registered_peer_mask();
            for round in 0..3 {
                failed_stage = format!("exchange-{round}");
                let timer = Instant::now();
                let width = WIDTHS[round];
                let neighbors = SENDERS.len();
                let incoming = STAGES[round + 1].1;
                let mut packed = vec![0.0; 60 * incoming];
                let mut received = vec![false; 60 * neighbors];
                for t in 0..60 {
                    packed[t * incoming..t * incoming + width]
                        .copy_from_slice(&hidden[t * width..(t + 1) * width]);
                }
                let cache_ready = exchange::update_round(window, round, &hidden);
                let nonce = (unsafe {
                    ((esp_idf_svc::sys::esp_random() as u64) << 32)
                        | esp_idf_svc::sys::esp_random() as u64
                } & 0x00ff_ffff_ffff_ffff)
                    | ((round as u64) << 56);
                exchange::begin_response_cycle(window, nonce);
                let mut requested = vec![0usize; neighbors];
                let mut queued = vec![0usize; neighbors];
                let mut attempts = vec![0usize; neighbors];
                let round_deadline = started
                    + Duration::from_millis(config::EXCHANGE_CUTOFF_MS * (round as u64 + 1) / 3)
                        .saturating_sub(age);
                let mut next_retry = Instant::now();
                while cache_ready && Instant::now() < round_deadline.min(deadline) {
                    if Instant::now() >= next_retry {
                        for n in 0..neighbors {
                            if !available[n] {
                                continue;
                            }
                            let missing: Vec<u16> = (0..60)
                                .filter(|t| !received[t * neighbors + n])
                                .map(|t| t as u16)
                                .collect();
                            if missing.is_empty() {
                                continue;
                            }
                            requested[n] = 60;
                            attempts[n] += 1;
                            if radio
                                .send(espnow::EspNowJob {
                                    target: config::NEIGHBORS[n].mac,
                                    payload: exchange::encode_request(
                                        config::DEVICE_ID,
                                        window,
                                        nonce,
                                        &missing,
                                    ),
                                    deadline: round_deadline.min(deadline),
                                })
                                .is_ok()
                            {
                                queued[n] += 1;
                            }
                        }
                        next_retry = Instant::now() + Duration::from_millis(2000);
                    }
                    for response in exchange::take_responses() {
                        if response.window_id != window || response.request_nonce != nonce {
                            continue;
                        }
                        let Some(n) = config::NEIGHBORS
                            .iter()
                            .position(|p| p.device_id == response.responder)
                        else {
                            continue;
                        };
                        if requested[n] == 0 {
                            continue;
                        }
                        for (i, t) in response.timesteps.iter().enumerate() {
                            let t = *t as usize;
                            let base = t * incoming + width * (1 + n);
                            packed[base..base + width]
                                .copy_from_slice(&response.hidden[i * width..(i + 1) * width]);
                            packed[t * incoming + width * (1 + neighbors) + n] = 1.0;
                            received[t * neighbors + n] = true;
                        }
                    }
                    if (0..neighbors)
                        .all(|n| !available[n] || (0..60).all(|t| received[t * neighbors + n]))
                    {
                        break;
                    }
                    esp_idf_hal::delay::FreeRtos::delay_ms(config::EXCHANGE_POLL_MS as u32);
                }
                exchange::end_response_cycle();
                let edges: Vec<_> = (0..neighbors).map(|n| serde_json::json!({"sender_node_index": SENDERS[n],
                    "possible_messages":60, "registered": available[n], "requested_messages":requested[n],
                    "request_attempts":attempts[n], "queued_requests":queued[n],
                    "received_messages":(0..60).filter(|t| received[t*neighbors+n]).count()})).collect();
                rounds.push(serde_json::json!({"round":round, "payload_width":width, "payload_bytes_per_message":width*4,
                    "cache_ready":cache_ready, "elapsed_us":timer.elapsed().as_micros() as u64, "edges":edges}));
                failed_stage = format!("graph-{round}");
                if Instant::now() >= cycle_deadline {
                    return Err("cycle deadline before graph stage".into());
                }
                let timer = Instant::now();
                hidden = self.invoke(round + 1, &packed, cycle_deadline)?;
                stages.push(serde_json::json!({"stage":failed_stage,"elapsed_us":timer.elapsed().as_micros() as u64}));
            }
            failed_stage = "classifier".into();
            if Instant::now() >= cycle_deadline {
                return Err("cycle deadline before classifier packing".into());
            }
            let mut packed = Vec::with_capacity(60 * STAGES[4].1);
            for t in 0..60 {
                packed.extend_from_slice(&h0[t * WIDTHS[0]..(t + 1) * WIDTHS[0]]);
                packed.extend_from_slice(&hidden[t * WIDTHS[1]..(t + 1) * WIDTHS[1]]);
            }
            let timer = Instant::now();
            let logits = self.invoke(4, &packed, cycle_deadline)?;
            stages.push(serde_json::json!({"stage":"classifier","elapsed_us":timer.elapsed().as_micros() as u64}));
            if Instant::now() >= cycle_deadline {
                return Err("cycle deadline exceeded; result discarded".into());
            }
            let decoded: Vec<usize> = logits
                .chunks_exact(4)
                .map(|row| (1..4).fold(0, |best, i| if row[i] > row[best] { i } else { best }))
                .collect();
            let names = ["NORMAL", "SPIKE", "DRIFT", "STUCK"];
            let decoded_labels: Vec<_> = decoded.iter().map(|i| names[*i]).collect();
            let received: usize = rounds
                .iter()
                .flat_map(|r| r["edges"].as_array().unwrap())
                .map(|e| e["received_messages"].as_u64().unwrap() as usize)
                .sum();
            let coverage = if received == 0 {
                "degraded_no_compatible_peer_payloads"
            } else if received == 3 * 60 * SENDERS.len() {
                "all_direct_neighbor_round_payloads_received"
            } else {
                "degraded_partial_round_payloads"
            };
            Ok(
                serde_json::json!({"device_id":config::DEVICE_ID,"type":"inference","model":"hifinet", "window_id":window,
                "window_ready_epoch_us":ready, "decoded_classes":decoded, "decoded_labels":decoded_labels,"emitted_labels":labels,
                "label":names[decoded[59]],"class":decoded[59], "coverage":coverage, "full_graph_verified":false,
                "decoder":"per_timestep_argmax_no_crf"}),
            )
        })();
        exchange::end_response_cycle();
        let succeeded = result.is_ok();
        if succeeded {
            telemetry.cycles_completed += 1;
        }
        let communication = communication_summary(&rounds, SENDERS, &WIDTHS);
        let stage_us = |name: &str| {
            stages
                .iter()
                .filter(|s| s["stage"].as_str() == Some(name))
                .map(|s| s["elapsed_us"].as_u64().unwrap_or(0))
                .sum::<u64>()
        };
        let aggregate_us: u64 = stages
            .iter()
            .filter(|s| s["stage"].as_str() != Some("temporal"))
            .map(|s| s["elapsed_us"].as_u64().unwrap_or(0))
            .sum();
        let exchange_us: u64 = rounds
            .iter()
            .map(|r| r["elapsed_us"].as_u64().unwrap_or(0))
            .sum();
        let decision_ready = succeeded.then(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_micros() as u64
        });
        telemetry.emit(publisher, serde_json::json!({"type":if succeeded {"benchmark"} else {"benchmark_status"},
            "model":"hifinet", "checkpoint_sha256":CHECKPOINT,"bundle_fingerprint":FINGERPRINT,"window_end_timestamp":window,
            "window_start_timestamp":window.saturating_sub(59*60),"prediction_policy":"window_all",
            "decision_ready_epoch_us":decision_ready,"window_ready_epoch_us":ready,"sample_interval_seconds":60,"window_size":60,
            "timing":{"request_inference_us":stage_us("temporal"),"aggregate_inference_us":aggregate_us,
                "exchange_wait_us":exchange_us,"cycle_us":started.elapsed().as_micros() as u64},
            "communication":communication,"rounds":rounds,"stages":stages,
            "failed_stage":if succeeded {None} else {Some(failed_stage)},"error":result.as_ref().err(),"memory":telemetry::memory()}));
        match result {
            Ok(payload) => {
                log::info!("[HIFINET] {}", payload);
                crate::publish_json(publisher, topic, payload);
            }
            Err(error) => log::error!("[HIFINET] window={} failed: {}", window, error),
        }
    }
}

fn communication_summary(
    rounds: &[serde_json::Value],
    senders: &[usize],
    widths: &[usize; 3],
) -> serde_json::Value {
    let edges: Vec<_> = senders
        .iter()
        .map(|sender| {
            let mut requested = 0u64;
            let mut received = 0u64;
            let mut requested_bytes = 0u64;
            let mut received_bytes = 0u64;
            for round in rounds {
                let bytes = round["payload_bytes_per_message"].as_u64().unwrap_or(0);
                for edge in round["edges"].as_array().into_iter().flatten() {
                    if edge["sender_node_index"].as_u64() != Some(*sender as u64) {
                        continue;
                    }
                    let req = edge["requested_messages"].as_u64().unwrap_or(0);
                    let rec = edge["received_messages"].as_u64().unwrap_or(0);
                    requested += req;
                    received += rec;
                    requested_bytes += req * bytes;
                    received_bytes += rec * bytes;
                }
            }
            serde_json::json!({"sender_node_index":sender,"possible_messages":180,
            "requested_messages":requested,"received_messages":received,
            "possible_payload_bytes":60*4*widths.iter().sum::<usize>(),
            "requested_payload_bytes":requested_bytes,"received_payload_bytes":received_bytes})
        })
        .collect();
    let total = |key: &str| {
        edges
            .iter()
            .map(|e| e[key].as_u64().unwrap_or(0))
            .sum::<u64>()
    };
    serde_json::json!({"purpose":"radio_energy_model_inputs","message_identity":"round_edge_timestep",
        "payload_bytes_per_round":widths.map(|w| w*4),"possible_messages":total("possible_messages"),
        "requested_messages":total("requested_messages"),"received_messages":total("received_messages"),
        "possible_payload_bytes":total("possible_payload_bytes"),"requested_payload_bytes":total("requested_payload_bytes"),
        "received_payload_bytes":total("received_payload_bytes"),"edges":edges})
}
