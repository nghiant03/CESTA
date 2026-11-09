//! Distributed node inference through the exported TFLite artifact.

use crate::tflite::StageRunner;
use std::time::Instant;

const WINDOW_SIZE: usize = 60;
const CLASS_COUNT: usize = 4;
const CLASS_NAMES: [&str; CLASS_COUNT] = ["NORMAL", "SPIKE", "DRIFT", "STUCK"];

static MODEL_DATA: &[u8] = include_bytes!("../../../model/model.tflite");
static MODEL_METADATA: &str = include_str!("../../../model/model.json");

pub struct NodeClassifier {
    runner: StageRunner,
    receiver_index: usize,
    hidden_size: usize,
    features_per_node: usize,
    neighbor_count: usize,
    input_width: usize,
    output_width: usize,
    request_threshold: f32,
    communication_mode: String,
    crf_transitions: Option<[[f32; CLASS_COUNT]; CLASS_COUNT]>,
    sample_interval_secs: u64,
    model_fingerprint: String,
    possible: Vec<bool>,
    samples: Vec<f32>,
    sample_count: usize,
    input: Vec<f32>,
    output: Vec<f32>,
}

impl NodeClassifier {
    pub fn new(
        tensor_arena_size: usize,
        receiver_index: usize,
        sender_indices: &[usize],
    ) -> Result<Self, String> {
        let (
            hidden_size,
            features_per_node,
            neighbor_count,
            input_width,
            output_width,
            request_threshold,
            communication_mode,
            crf_transitions,
            sample_interval_secs,
            model_fingerprint,
        ) = parse_metadata(receiver_index, sender_indices)?;
        let mut runner = StageRunner::new(tensor_arena_size)?;
        runner.load(MODEL_DATA, input_width, output_width)?;
        Ok(Self {
            runner,
            receiver_index,
            hidden_size,
            features_per_node,
            neighbor_count,
            input_width,
            output_width,
            request_threshold,
            communication_mode,
            crf_transitions,
            sample_interval_secs,
            model_fingerprint,
            possible: vec![true; neighbor_count],
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

    pub fn sample_interval_secs(&self) -> u64 {
        self.sample_interval_secs
    }

    pub fn model_fingerprint(&self) -> &str {
        &self.model_fingerprint
    }

    pub fn reset_window(&mut self) {
        self.sample_count = 0;
        self.samples.fill(0.0);
    }

    pub fn set_neighbor_available(
        &mut self,
        neighbor: usize,
        available: bool,
    ) -> Result<(), String> {
        let slot = self
            .possible
            .get_mut(neighbor)
            .ok_or_else(|| "neighbor index is out of range".to_owned())?;
        *slot = available;
        Ok(())
    }

    pub fn push_temperature(&mut self, temperature: f32) -> bool {
        if !temperature.is_finite() {
            self.reset_window();
            return false;
        }
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
        NeighborSlots::new(
            WINDOW_SIZE,
            self.neighbor_count,
            self.hidden_size,
            self.features_per_node,
        )
    }

    pub fn predict(&mut self, slots: Option<&NeighborSlots>) -> Result<NodePass, String> {
        let started = Instant::now();
        if self.sample_count != WINDOW_SIZE {
            return Err("inference requires a complete contiguous window".to_owned());
        }
        if let Some(slots) = slots {
            if slots.window_size != WINDOW_SIZE
                || slots.neighbor_count != self.neighbor_count
                || slots.hidden_size != self.hidden_size
                || slots.features_per_node != self.features_per_node
                || slots.payload.len()
                    != WINDOW_SIZE
                        * self.neighbor_count
                        * (self.hidden_size + self.features_per_node)
                || slots.received.len() != WINDOW_SIZE * self.neighbor_count
                || slots.payload.iter().any(|v| !v.is_finite())
                || slots.received.iter().any(|v| *v != 0.0 && *v != 1.0)
            {
                return Err("invalid neighbor slots".to_owned());
            }
        }
        self.build_input(slots);
        self.runner.predict(&self.input, &mut self.output)?;
        let mut pass = NodePass {
            logits: vec![0.0; WINDOW_SIZE * CLASS_COUNT],
            decoded: Vec::new(),
            probabilities: vec![0.0; WINDOW_SIZE * CLASS_COUNT],
            hidden: vec![0.0; WINDOW_SIZE * self.hidden_size],
            request: vec![0.0; WINDOW_SIZE * self.neighbor_count],
            elapsed_ms: started.elapsed().as_millis(),
        };
        for timestep in 0..WINDOW_SIZE {
            let row = timestep * self.output_width;
            let classes = timestep * CLASS_COUNT;
            pass.logits[classes..classes + CLASS_COUNT]
                .copy_from_slice(&self.output[row..row + CLASS_COUNT]);
            let probabilities = softmax(&self.output[row..row + CLASS_COUNT]);
            pass.probabilities[classes..classes + CLASS_COUNT].copy_from_slice(&probabilities);
            let hidden = timestep * self.hidden_size;
            pass.hidden[hidden..hidden + self.hidden_size].copy_from_slice(
                &self.output[row + CLASS_COUNT..row + CLASS_COUNT + self.hidden_size],
            );
            let requests = timestep * self.neighbor_count;
            pass.request[requests..requests + self.neighbor_count].copy_from_slice(
                &self.output[row + CLASS_COUNT + self.hidden_size..row + self.output_width],
            );
        }
        pass.decoded = decode_sequence(&pass.logits, self.crf_transitions.as_ref())?;
        pass.elapsed_ms = started.elapsed().as_millis();
        Ok(pass)
    }

    pub fn threshold_requests(&self, request: &[f32]) -> Vec<Vec<u16>> {
        (0..self.neighbor_count)
            .map(|neighbor| {
                (0..WINDOW_SIZE)
                    .filter(|timestep| {
                        self.possible[neighbor]
                            && request[timestep * self.neighbor_count + neighbor]
                                >= self.request_threshold
                    })
                    .map(|timestep| timestep as u16)
                    .collect()
            })
            .collect()
    }

    pub fn diagnosis(&self, pass: &NodePass) -> Diagnosis {
        let offset = (WINDOW_SIZE - 1) * CLASS_COUNT;
        let probabilities = pass.probabilities[offset..offset + CLASS_COUNT].to_vec();
        let class = pass.decoded[WINDOW_SIZE - 1];
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
                self.input[cursor + neighbor] = if self.possible[neighbor] { 1.0 } else { 0.0 };
            }
            cursor += neighbors;
            for neighbor in 0..neighbors {
                self.input[cursor + neighbor] = match slots {
                    Some(slots) if self.possible[neighbor] => {
                        slots.received[timestep * neighbors + neighbor]
                    }
                    Some(_) => 0.0,
                    None => 0.0,
                };
            }
        }
    }
}

pub struct NodePass {
    pub logits: Vec<f32>,
    pub decoded: Vec<usize>,
    pub probabilities: Vec<f32>,
    pub hidden: Vec<f32>,
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
    pub fn new(
        window_size: usize,
        neighbor_count: usize,
        hidden_size: usize,
        features_per_node: usize,
    ) -> Self {
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

    pub fn fill(
        &mut self,
        timestep: usize,
        neighbor: usize,
        hidden: &[f32],
        features: &[f32],
    ) -> bool {
        if timestep >= self.window_size
            || neighbor >= self.neighbor_count
            || hidden.len() != self.hidden_size
            || features.len() != self.features_per_node
            || hidden
                .iter()
                .chain(features)
                .any(|value| !value.is_finite())
        {
            return false;
        }
        let index = timestep * self.neighbor_count + neighbor;
        let newly = self.received[index] == 0.0;
        let base = timestep * self.neighbor_count * (self.hidden_size + self.features_per_node)
            + neighbor * (self.hidden_size + self.features_per_node);
        self.payload[base..base + self.hidden_size].copy_from_slice(hidden);
        self.payload[base + self.hidden_size..base + self.hidden_size + self.features_per_node]
            .copy_from_slice(features);
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
    usize,  // hidden_size
    usize,  // features_per_node
    usize,  // neighbor_count
    usize,  // input_width
    usize,  // output_width
    f32,    // request_threshold
    String, // communication_mode
    Option<[[f32; CLASS_COUNT]; CLASS_COUNT]>,
    u64,
    String,
);

fn parse_metadata(receiver_index: usize, sender_indices: &[usize]) -> Result<Metadata, String> {
    let metadata: serde_json::Value = serde_json::from_str(MODEL_METADATA)
        .map_err(|error| format!("invalid model metadata: {}", error))?;
    if json_str(&metadata, "format")? != "cesta-tflite-micro-v3" {
        return Err(
            "firmware requires a v3 raw-logit export, legacy exports are unsupported".to_owned(),
        );
    }
    if metadata.get("output_layout")
        != Some(&serde_json::json!([
            "class_logits",
            "local_hidden",
            "request_mask"
        ]))
    {
        return Err(
            "model output_layout must contain raw class_logits, local_hidden, request_mask"
                .to_owned(),
        );
    }
    if json_str(&metadata, "temporal_backend")? != "gru" {
        return Err("firmware supports only the GRU temporal backend".to_owned());
    }
    let bidirectional = json_bool(&metadata, "gru_bidirectional")?;
    if !bidirectional || json_u64(&metadata, "gru_num_layers")? != 2 {
        return Err("firmware requires a two-layer bidirectional GRU".to_owned());
    }
    json_bool(&metadata, "standardize_window_features")?;
    let sample_interval_secs = json_u64(&metadata, "sample_interval_secs")?;
    if sample_interval_secs != 60 {
        return Err("firmware requires a 60-second sample interval".to_owned());
    }
    let model_fingerprint = json_str(&metadata, "source_checkpoint_sha256")?;
    if model_fingerprint.len() != 64
        || !model_fingerprint
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(
            "source_checkpoint_sha256 must contain exactly 64 hexadecimal digits".to_owned(),
        );
    }
    let model_fingerprint = model_fingerprint.to_ascii_lowercase();
    let crf_transitions = parse_transitions(&metadata)?;
    if metadata
        .get("trained_checkpoint")
        .and_then(serde_json::Value::as_bool)
        != Some(true)
    {
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
    if metadata_senders
        != sender_indices
            .iter()
            .map(|index| *index as u64)
            .collect::<Vec<_>>()
    {
        return Err(
            "model sender_indices do not match the configured NEIGHBORS node indices".to_owned(),
        );
    }
    let neighbor_count = json_u64(&metadata, "neighbor_count")? as usize;
    if neighbor_count != metadata_senders.len() {
        return Err("model neighbor_count does not match sender_indices".to_owned());
    }
    let hidden_size = json_u64(&metadata, "hidden_size")? as usize;
    if hidden_size == 0 || (bidirectional && hidden_size % 2 != 0) || neighbor_count == 0 {
        return Err("invalid hidden width or neighbor count".to_owned());
    }
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
    Ok((
        hidden_size,
        features_per_node,
        neighbor_count,
        input_width,
        output_width,
        request_threshold,
        communication_mode,
        crf_transitions,
        sample_interval_secs,
        model_fingerprint,
    ))
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
            value
                .as_u64()
                .ok_or_else(|| format!("model metadata array '{}' must contain integers", key))
        })
        .collect()
}

fn json_bool(metadata: &serde_json::Value, key: &str) -> Result<bool, String> {
    metadata
        .get(key)
        .and_then(serde_json::Value::as_bool)
        .ok_or_else(|| format!("model metadata is missing boolean field '{}'", key))
}

fn parse_transitions(
    metadata: &serde_json::Value,
) -> Result<Option<[[f32; CLASS_COUNT]; CLASS_COUNT]>, String> {
    let enabled = json_bool(metadata, "use_crf")?;
    let value = metadata
        .get("crf_transitions")
        .ok_or_else(|| "missing crf_transitions".to_owned())?;
    if !enabled {
        if !value.is_null() {
            return Err("disabled CRF requires null crf_transitions".to_owned());
        }
        return Ok(None);
    }
    let rows = value
        .as_array()
        .filter(|rows| rows.len() == CLASS_COUNT)
        .ok_or_else(|| "crf_transitions must be a 4x4 matrix".to_owned())?;
    let mut transitions = [[0.0; CLASS_COUNT]; CLASS_COUNT];
    for (previous, row) in rows.iter().enumerate() {
        let row = row
            .as_array()
            .filter(|row| row.len() == CLASS_COUNT)
            .ok_or_else(|| "crf_transitions must be a 4x4 matrix".to_owned())?;
        for (next, value) in row.iter().enumerate() {
            let value = value
                .as_f64()
                .ok_or_else(|| "CRF scores must be numbers".to_owned())?
                as f32;
            if !value.is_finite() {
                return Err("CRF scores must be finite float32 values".to_owned());
            }
            transitions[previous][next] = value;
        }
    }
    Ok(Some(transitions))
}

fn argmax(values: &[f32]) -> usize {
    let mut best = 0;
    for index in 1..values.len() {
        if values[index] > values[best] {
            best = index;
        }
    }
    best
}

fn softmax(logits: &[f32]) -> [f32; CLASS_COUNT] {
    let maximum = logits[argmax(logits)];
    let mut probabilities = [0.0; CLASS_COUNT];
    for index in 0..CLASS_COUNT {
        probabilities[index] = (logits[index] - maximum).exp();
    }
    let sum: f32 = probabilities.iter().sum();
    for value in &mut probabilities {
        *value /= sum;
    }
    probabilities
}

fn decode_sequence(
    logits: &[f32],
    transitions: Option<&[[f32; CLASS_COUNT]; CLASS_COUNT]>,
) -> Result<Vec<usize>, String> {
    if logits.is_empty() || logits.len() % CLASS_COUNT != 0 || logits.iter().any(|v| !v.is_finite())
    {
        return Err("invalid sequence emissions".to_owned());
    }
    let Some(transitions) = transitions else {
        return Ok(logits.chunks_exact(CLASS_COUNT).map(argmax).collect());
    };
    let length = logits.len() / CLASS_COUNT;
    let mut score = [0.0; CLASS_COUNT];
    score.copy_from_slice(&logits[..CLASS_COUNT]);
    let mut backpointers = vec![[0; CLASS_COUNT]; length];
    for timestep in 1..length {
        let mut next_score = [0.0; CLASS_COUNT];
        for next in 0..CLASS_COUNT {
            let mut candidates = [0.0; CLASS_COUNT];
            for previous in 0..CLASS_COUNT {
                candidates[previous] = score[previous] + transitions[previous][next];
            }
            let best = argmax(&candidates);
            backpointers[timestep][next] = best;
            next_score[next] = candidates[best] + logits[timestep * CLASS_COUNT + next];
        }
        if next_score.iter().any(|v| !v.is_finite()) {
            return Err("CRF score overflow".to_owned());
        }
        score = next_score;
    }
    let mut decoded = vec![0; length];
    decoded[length - 1] = argmax(&score);
    for timestep in (1..length).rev() {
        decoded[timestep - 1] = backpointers[timestep][decoded[timestep]];
    }
    Ok(decoded)
}
