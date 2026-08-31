//! Distributed CESTA node inference through the exported TFLite artifact.
//!
//! The embedded model is a `--target node` export from
//! `scripts/export_cesta_firmware.py`. Each pass consumes the local window plus
//! per-neighbor hidden-state payloads and returns per-timestep class
//! probabilities, the local hidden state, and receiver-local request
//! probabilities for every possible sender.

use std::ffi::{CStr, c_char, c_int, c_uchar, c_void};
use std::ptr::NonNull;
use std::time::Instant;

const WINDOW_SIZE: usize = 60;
const CLASS_COUNT: usize = 4;
const CLASS_NAMES: [&str; CLASS_COUNT] = ["NORMAL", "SPIKE", "DRIFT", "STUCK"];

static MODEL_DATA: &[u8] = include_bytes!("../model/model.tflite");
static MODEL_METADATA: &str = include_str!("../model/model.json");

unsafe extern "C" {
    fn cesta_tflite_create(
        model_data: *const c_uchar,
        model_size: usize,
        tensor_arena_size: usize,
    ) -> *mut c_void;
    fn cesta_tflite_destroy(classifier: *mut c_void);
    fn cesta_tflite_last_error(classifier: *const c_void) -> *const c_char;
    fn cesta_tflite_predict(
        classifier: *mut c_void,
        input: *const f32,
        input_count: usize,
        output: *mut f32,
        output_count: usize,
    ) -> c_int;
}

pub struct NodeClassifier {
    inner: NonNull<c_void>,
    receiver_index: usize,
    hidden_size: usize,
    features_per_node: usize,
    neighbor_count: usize,
    input_width: usize,
    output_width: usize,
    request_threshold: f32,
    communication_mode: String,
    samples: Vec<f32>,
    sample_count: usize,
    input: Vec<f32>,
    output: Vec<f32>,
}

impl NodeClassifier {
    pub fn new(tensor_arena_size: usize, receiver_index: usize, sender_indices: &[usize]) -> Result<Self, String> {
        let (hidden_size, features_per_node, neighbor_count, input_width, output_width, request_threshold, communication_mode) =
            parse_metadata(receiver_index, sender_indices)?;
        let inner = unsafe {
            cesta_tflite_create(MODEL_DATA.as_ptr(), MODEL_DATA.len(), tensor_arena_size)
        };
        let inner =
            NonNull::new(inner).ok_or_else(|| "failed to allocate TensorFlow Lite classifier".to_owned())?;
        let error = unsafe { cesta_tflite_last_error(inner.as_ptr()) };
        if !error.is_null() {
            let message = unsafe { CStr::from_ptr(error) }
                .to_string_lossy()
                .into_owned();
            unsafe { cesta_tflite_destroy(inner.as_ptr()) };
            return Err(message);
        }
        Ok(Self {
            inner,
            receiver_index,
            hidden_size,
            features_per_node,
            neighbor_count,
            input_width,
            output_width,
            request_threshold,
            communication_mode,
            samples: vec![0.0; WINDOW_SIZE * features_per_node],
            sample_count: 0,
            input: vec![0.0; WINDOW_SIZE * input_width],
            output: vec![0.0; WINDOW_SIZE * output_width],
        })
    }

    pub fn window_size(&self) -> usize {
        WINDOW_SIZE
    }

    pub fn receiver_index(&self) -> usize {
        self.receiver_index
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn features_per_node(&self) -> usize {
        self.features_per_node
    }

    pub fn neighbor_count(&self) -> usize {
        self.neighbor_count
    }

    pub fn communication_mode(&self) -> &str {
        &self.communication_mode
    }

    /// Append a temperature sample to the sliding window and report whether a
    /// full window is available.
    pub fn push_temperature(&mut self, temperature: f32) -> bool {
        let features = self.features_per_node;
        if self.sample_count < WINDOW_SIZE {
            self.samples[self.sample_count * features] = temperature;
            self.sample_count += 1;
            if self.sample_count < WINDOW_SIZE {
                return false;
            }
        } else {
            self.samples.copy_within(features.., 0);
            let last = WINDOW_SIZE * features;
            self.samples[last - features..].fill(temperature);
        }
        true
    }

    pub fn features(&self) -> &[f32] {
        &self.samples
    }

    pub fn slots(&self) -> NeighborSlots {
        NeighborSlots::new(WINDOW_SIZE, self.neighbor_count, self.hidden_size, self.features_per_node)
    }

    /// Run one inference pass. `slots` carries received neighbor payloads; pass
    /// `None` for the receiver-local request pass without neighbor context.
    pub fn predict(&mut self, slots: Option<&NeighborSlots>) -> Result<NodePass, String> {
        let started = Instant::now();
        self.build_input(slots);
        let status = unsafe {
            cesta_tflite_predict(
                self.inner.as_ptr(),
                self.input.as_ptr(),
                self.input.len(),
                self.output.as_mut_ptr(),
                self.output.len(),
            )
        };
        if status != 0 {
            let error = unsafe { cesta_tflite_last_error(self.inner.as_ptr()) };
            return Err(if error.is_null() {
                "TensorFlow Lite inference failed".to_owned()
            } else {
                unsafe { CStr::from_ptr(error) }
                    .to_string_lossy()
                    .into_owned()
            });
        }

        let mut pass = NodePass {
            probabilities: vec![0.0; WINDOW_SIZE * CLASS_COUNT],
            hidden: vec![0.0; WINDOW_SIZE * self.hidden_size],
            request: vec![0.0; WINDOW_SIZE * self.neighbor_count],
            elapsed_ms: started.elapsed().as_millis(),
        };
        for timestep in 0..WINDOW_SIZE {
            let row = timestep * self.output_width;
            let classes = timestep * CLASS_COUNT;
            pass.probabilities[classes..classes + CLASS_COUNT]
                .copy_from_slice(&self.output[row..row + CLASS_COUNT]);
            let hidden = timestep * self.hidden_size;
            pass.hidden[hidden..hidden + self.hidden_size]
                .copy_from_slice(&self.output[row + CLASS_COUNT..row + CLASS_COUNT + self.hidden_size]);
            let requests = timestep * self.neighbor_count;
            pass.request[requests..requests + self.neighbor_count].copy_from_slice(
                &self.output[row + CLASS_COUNT + self.hidden_size..row + self.output_width],
            );
        }
        Ok(pass)
    }

    /// Threshold receiver-local request probabilities into per-neighbor
    /// requested timestep indices, mirroring evaluation semantics.
    pub fn threshold_requests(&self, request: &[f32]) -> Vec<Vec<u16>> {
        (0..self.neighbor_count)
            .map(|neighbor| {
                (0..WINDOW_SIZE)
                    .filter(|timestep| request[timestep * self.neighbor_count + neighbor] >= self.request_threshold)
                    .map(|timestep| timestep as u16)
                    .collect()
            })
            .collect()
    }

    pub fn diagnosis(&self, pass: &NodePass) -> Diagnosis {
        let offset = (WINDOW_SIZE - 1) * CLASS_COUNT;
        let probabilities = pass.probabilities[offset..offset + CLASS_COUNT].to_vec();
        let mut class = 0;
        for index in 1..CLASS_COUNT {
            if probabilities[index] > probabilities[class] {
                class = index;
            }
        }
        Diagnosis {
            class,
            label: CLASS_NAMES[class],
            confidence: probabilities[class],
            probabilities,
        }
    }

    fn build_input(&mut self, slots: Option<&NeighborSlots>) {
        let features = self.features_per_node;
        let neighbors = self.neighbor_count;
        let payload_width = self.hidden_size + features;
        for timestep in 0..WINDOW_SIZE {
            let row = timestep * self.input_width;
            self.input[row..row + features]
                .copy_from_slice(&self.samples[timestep * features..(timestep + 1) * features]);
            let mut cursor = row + features;
            match slots {
                Some(slots) => {
                    let base = timestep * neighbors * payload_width;
                    for neighbor in 0..neighbors {
                        let source = base + neighbor * payload_width;
                        self.input[cursor..cursor + payload_width]
                            .copy_from_slice(&slots.payload[source..source + payload_width]);
                        cursor += payload_width;
                    }
                }
                None => {
                    self.input[cursor..cursor + neighbors * payload_width].fill(0.0);
                    cursor += neighbors * payload_width;
                }
            }
            for neighbor in 0..neighbors {
                self.input[cursor + neighbor] = 1.0;
            }
            cursor += neighbors;
            for neighbor in 0..neighbors {
                self.input[cursor + neighbor] = match slots {
                    Some(slots) => slots.received[timestep * neighbors + neighbor],
                    None => 0.0,
                };
            }
        }
    }
}

impl Drop for NodeClassifier {
    fn drop(&mut self) {
        unsafe { cesta_tflite_destroy(self.inner.as_ptr()) };
    }
}

pub struct NodePass {
    /// Per-timestep class probabilities, `window_size * class_count`.
    pub probabilities: Vec<f32>,
    /// Local hidden states for this window, `window_size * hidden_size`.
    pub hidden: Vec<f32>,
    /// Receiver-local request probabilities, `window_size * neighbor_count`.
    pub request: Vec<f32>,
    pub elapsed_ms: u128,
}

pub struct Diagnosis {
    pub class: usize,
    pub label: &'static str,
    pub confidence: f32,
    pub probabilities: Vec<f32>,
}

pub struct NeighborSlots {
    window_size: usize,
    neighbor_count: usize,
    hidden_size: usize,
    features_per_node: usize,
    pub payload: Vec<f32>,
    pub received: Vec<f32>,
}

impl NeighborSlots {
    pub fn new(window_size: usize, neighbor_count: usize, hidden_size: usize, features_per_node: usize) -> Self {
        let payload_width = neighbor_count * (hidden_size + features_per_node);
        Self {
            window_size,
            neighbor_count,
            hidden_size,
            features_per_node,
            payload: vec![0.0; window_size * payload_width],
            received: vec![0.0; window_size * neighbor_count],
        }
    }

    /// Fill a neighbor payload slot; returns true when the slot was newly
    /// received, false when it was already received or the input is invalid.
    pub fn fill(&mut self, timestep: usize, neighbor: usize, hidden: &[f32], features: &[f32]) -> bool {
        if timestep >= self.window_size
            || neighbor >= self.neighbor_count
            || hidden.len() != self.hidden_size
            || features.len() != self.features_per_node
        {
            return false;
        }
        let index = timestep * self.neighbor_count + neighbor;
        let newly = self.received[index] == 0.0;
        let base = timestep * self.neighbor_count * (self.hidden_size + self.features_per_node)
            + neighbor * (self.hidden_size + self.features_per_node);
        self.payload[base..base + self.hidden_size].copy_from_slice(hidden);
        self.payload[base + self.hidden_size..base + self.hidden_size + self.features_per_node].copy_from_slice(features);
        self.received[index] = 1.0;
        newly
    }

    pub fn is_received(&self, timestep: usize, neighbor: usize) -> bool {
        timestep < self.window_size
            && neighbor < self.neighbor_count
            && self.received[timestep * self.neighbor_count + neighbor] != 0.0
    }

    pub fn received_count(&self) -> usize {
        self.received.iter().filter(|value| **value != 0.0).count()
    }
}

type Metadata = (
    usize, // hidden_size
    usize, // features_per_node
    usize, // neighbor_count
    usize, // input_width
    usize, // output_width
    f32,   // request_threshold
    String, // communication_mode
);

fn parse_metadata(receiver_index: usize, sender_indices: &[usize]) -> Result<Metadata, String> {
    let metadata: serde_json::Value =
        serde_json::from_str(MODEL_METADATA).map_err(|error| format!("invalid model metadata: {}", error))?;
    if metadata.get("trained_checkpoint").and_then(serde_json::Value::as_bool) != Some(true) {
        return Err("firmware model must be exported from a trained CESTA checkpoint".to_owned());
    }
    if metadata.get("target").and_then(serde_json::Value::as_str) != Some("node") {
        return Err("firmware requires a distributed node export: run scripts/export_cesta_firmware.py with --target node".to_owned());
    }
    if json_u64(&metadata, "window_size")? as usize != WINDOW_SIZE {
        return Err("firmware only supports 60-sample windows".to_owned());
    }
    if json_u64(&metadata, "num_classes")? as usize != CLASS_COUNT {
        return Err("firmware only supports NORMAL, SPIKE, DRIFT, and STUCK classes".to_owned());
    }
    let features_per_node = json_u64(&metadata, "features_per_node")? as usize;
    if features_per_node != 1 {
        return Err("firmware export requires one feature per node".to_owned());
    }
    if json_u64(&metadata, "receiver_index")? as usize != receiver_index {
        return Err("model receiver_index does not match config::NODE_INDEX".to_owned());
    }
    let metadata_senders = json_u64_slice(&metadata, "sender_indices")?;
    if metadata_senders != sender_indices.iter().map(|index| *index as u64).collect::<Vec<_>>() {
        return Err("model sender_indices do not match the configured NEIGHBORS node indices".to_owned());
    }
    let neighbor_count = json_u64(&metadata, "neighbor_count")? as usize;
    if neighbor_count != metadata_senders.len() {
        return Err("model neighbor_count does not match sender_indices".to_owned());
    }
    let hidden_size = json_u64(&metadata, "hidden_size")? as usize;
    let input_width =
        features_per_node + neighbor_count * (hidden_size + features_per_node) + 2 * neighbor_count;
    let output_width = CLASS_COUNT + hidden_size + neighbor_count;
    let input_shape = json_u64_slice(&metadata, "input_shape")?;
    let output_shape = json_u64_slice(&metadata, "output_shape")?;
    if input_shape != [1, WINDOW_SIZE as u64, input_width as u64] {
        return Err("model input_shape does not match the node deployment contract".to_owned());
    }
    if output_shape != [1, WINDOW_SIZE as u64, output_width as u64] {
        return Err("model output_shape does not match the node deployment contract".to_owned());
    }
    let request_threshold = json_f64(&metadata, "request_threshold")? as f32;
    if !(0.0..=1.0).contains(&request_threshold) {
        return Err("model request_threshold must be in [0, 1]".to_owned());
    }
    let communication_mode = json_str(&metadata, "communication_mode")?.to_owned();
    if communication_mode != "dense" && communication_mode != "gumbel_request" {
        return Err("node export must use dense or gumbel_request communication".to_owned());
    }
    Ok((hidden_size, features_per_node, neighbor_count, input_width, output_width, request_threshold, communication_mode))
}

fn json_u64(metadata: &serde_json::Value, key: &str) -> Result<u64, String> {
    metadata
        .get(key)
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| format!("model metadata is missing integer field '{}'", key))
}

fn json_f64(metadata: &serde_json::Value, key: &str) -> Result<f64, String> {
    metadata
        .get(key)
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| format!("model metadata is missing number field '{}'", key))
}

fn json_str<'a>(metadata: &'a serde_json::Value, key: &str) -> Result<&'a str, String> {
    metadata
        .get(key)
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| format!("model metadata is missing string field '{}'", key))
}

fn json_u64_slice(metadata: &serde_json::Value, key: &str) -> Result<Vec<u64>, String> {
    let values = metadata
        .get(key)
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| format!("model metadata is missing array field '{}'", key))?;
    values
        .iter()
        .map(|value| {
            value.as_u64().ok_or_else(|| format!("model metadata array '{}' must contain integers", key))
        })
        .collect()
}
