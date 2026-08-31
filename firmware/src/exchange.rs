//! Binary MQTT neighbor exchange protocol for distributed CESTA node inference.
//!
//! The protocol preserves CESTA's receiver-local request, neighbor-response
//! contract: a receiver publishes thresholded per-timestep requests to each
//! requested sender's mailbox, and senders answer from their most recent cached
//! hidden state for the requested timesteps only. Responses carry the
//! receiver's window id so late responses are dropped; senders serve windows
//! from their own latest inference pass, so payload alignment is approximate
//! when devices sample out of phase.
//!
//! Wire format (little-endian):
//! - request: `CESTR` | version | id_len | requester id | window_id u64 |
//!   count u16 | timesteps u16[]
//! - response: `CESTP` | version | id_len | responder id | window_id u64 |
//!   count u16 | timesteps u16[] | hidden f32[] | features f32[]
//!
//! A response with count 0 explicitly reports that the sender has no cached
//! window available.

use std::collections::VecDeque;
use std::sync::{Mutex, OnceLock};

use log::{debug, info, warn};

use crate::config;

const REQUEST_MAGIC: [u8; 5] = *b"CESTR";
const RESPONSE_MAGIC: [u8; 5] = *b"CESTP";
const PROTOCOL_VERSION: u8 = 1;
const MAX_DEVICE_ID: usize = 32;
const MAX_TIMESTEPS: usize = 60;

struct SharedState {
    window_size: usize,
    hidden_size: usize,
    features_per_node: usize,
    cache: Mutex<HiddenCache>,
    inbox: Mutex<Inbox>,
}

struct HiddenCache {
    window_id: u64,
    ready: bool,
    hidden: Vec<f32>,
    features: Vec<f32>,
}

struct Inbox {
    requests: VecDeque<Request>,
    responses: Vec<Response>,
}

static STATE: OnceLock<SharedState> = OnceLock::new();

#[derive(Clone)]
pub struct Request {
    pub requester: String,
    pub window_id: u64,
    pub timesteps: Vec<u16>,
}

#[derive(Clone)]
pub struct Response {
    pub responder: String,
    pub window_id: u64,
    pub timesteps: Vec<u16>,
    pub hidden: Vec<f32>,
    pub features: Vec<f32>,
}

pub fn init(window_size: usize, hidden_size: usize, features_per_node: usize) {
    let _ = STATE.set(SharedState {
        window_size,
        hidden_size,
        features_per_node,
        cache: Mutex::new(HiddenCache {
            window_id: 0,
            ready: false,
            hidden: vec![0.0; window_size * hidden_size],
            features: vec![0.0; window_size * features_per_node],
        }),
        inbox: Mutex::new(Inbox {
            requests: VecDeque::new(),
            responses: Vec::new(),
        }),
    });
}

fn state() -> Option<&'static SharedState> {
    STATE.get()
}

/// Whether the exchange was initialized with a loaded node model.
pub fn ready() -> bool {
    STATE.get().is_some()
}

pub fn request_topic(device_id: &str) -> String {
    format!("{}{}/request", config::EXCHANGE_TOPIC_PREFIX, device_id)
}

pub fn response_topic(device_id: &str) -> String {
    format!("{}{}/response", config::EXCHANGE_TOPIC_PREFIX, device_id)
}

/// Route a received MQTT message into the exchange; drops unknown frames.
pub fn handle_message(topic: &str, data: &[u8]) {
    if topic == request_topic(config::DEVICE_ID) {
        if let Some(request) = decode_request(data) {
            info!(
                "[EXCHANGE] request from {} for window {} on {} timesteps",
                request.requester,
                request.window_id,
                request.timesteps.len()
            );
            if let Some(state) = state()
                && let Ok(mut inbox) = state.inbox.lock()
            {
                inbox.requests.push_back(request);
            }
        } else {
            warn!("[EXCHANGE] dropped malformed request frame");
        }
    } else if topic == response_topic(config::DEVICE_ID) {
        if let Some(response) = decode_response(data) {
            debug!(
                "[EXCHANGE] response from {} for window {} on {} timesteps",
                response.responder,
                response.window_id,
                response.timesteps.len()
            );
            if let Some(state) = state()
                && let Ok(mut inbox) = state.inbox.lock()
            {
                inbox.responses.push(response);
            }
        } else {
            warn!("[EXCHANGE] dropped malformed response frame");
        }
    }
}

/// Publish the latest local hidden state and features for request responses.
/// Returns the window id the cached state is valid for.
pub fn update_cache(hidden: &[f32], features: &[f32]) -> u64 {
    let Some(state) = state() else {
        return 0;
    };
    let Ok(mut cache) = state.cache.lock() else {
        return cache_window_id(state);
    };
    if hidden.len() != cache.hidden.len() || features.len() != cache.features.len() {
        warn!("[EXCHANGE] hidden cache update ignored: payload size mismatch");
        return cache.window_id;
    }
    cache.window_id += 1;
    cache.hidden.copy_from_slice(hidden);
    cache.features.copy_from_slice(features);
    cache.ready = true;
    cache.window_id
}

fn cache_window_id(state: &SharedState) -> u64 {
    state
        .cache
        .lock()
        .map(|cache| cache.window_id)
        .unwrap_or(0)
}

/// Answer pending requests from the cached hidden state; returns response
/// `(topic, payload)` pairs for the MQTT worker to publish.
pub fn serve_pending_requests() -> Vec<(String, Vec<u8>)> {
    let Some(state) = state() else {
        return Vec::new();
    };
    let requests = {
        let Ok(mut inbox) = state.inbox.lock() else {
            return Vec::new();
        };
        inbox.requests.drain(..).collect::<Vec<_>>()
    };
    if requests.is_empty() {
        return Vec::new();
    }
    let cache = state.cache.lock();
    let Ok(cache) = cache else {
        return requests
            .into_iter()
            .map(|request| {
                (
                    response_topic(&request.requester),
                    encode_response(config::DEVICE_ID, request.window_id, &[], &[], &[]),
                )
            })
            .collect();
    };
    requests
        .into_iter()
        .map(|request| {
            let payload = if cache.ready && !request.timesteps.is_empty() {
                let mut hidden = Vec::with_capacity(request.timesteps.len() * state.hidden_size);
                let mut features = Vec::with_capacity(request.timesteps.len() * state.features_per_node);
                for timestep in &request.timesteps {
                    let timestep = *timestep as usize;
                    if timestep >= state.window_size {
                        continue;
                    }
                    hidden.extend_from_slice(&cache.hidden[timestep * state.hidden_size..(timestep + 1) * state.hidden_size]);
                    features.extend_from_slice(
                        &cache.features[timestep * state.features_per_node..(timestep + 1) * state.features_per_node],
                    );
                }
                encode_response(
                    config::DEVICE_ID,
                    request.window_id,
                    &request.timesteps,
                    &hidden,
                    &features,
                )
            } else {
                encode_response(config::DEVICE_ID, request.window_id, &[], &[], &[])
            };
            (response_topic(&request.requester), payload)
        })
        .collect()
}

/// Drain collected responses for the MQTT exchange; late or foreign-window
/// responses are dropped by the caller via window-id matching.
pub fn take_responses() -> Vec<Response> {
    let Some(state) = state() else {
        return Vec::new();
    };
    let Ok(mut inbox) = state.inbox.lock() else {
        return Vec::new();
    };
    std::mem::take(&mut inbox.responses)
}

pub fn encode_request(requester: &str, window_id: u64, timesteps: &[u16]) -> Vec<u8> {
    let mut frame = Vec::with_capacity(REQUEST_MAGIC.len() + 11 + requester.len() + timesteps.len() * 2);
    frame.extend_from_slice(&REQUEST_MAGIC);
    frame.push(PROTOCOL_VERSION);
    frame.push(requester.len() as u8);
    frame.extend_from_slice(requester.as_bytes());
    frame.extend_from_slice(&window_id.to_le_bytes());
    frame.extend_from_slice(&(timesteps.len() as u16).to_le_bytes());
    for timestep in timesteps {
        frame.extend_from_slice(&timestep.to_le_bytes());
    }
    frame
}

pub fn encode_response(
    responder: &str,
    window_id: u64,
    timesteps: &[u16],
    hidden: &[f32],
    features: &[f32],
) -> Vec<u8> {
    let mut frame = Vec::with_capacity(RESPONSE_MAGIC.len() + 11 + responder.len() + timesteps.len() * 6 + hidden.len() * 4);
    frame.extend_from_slice(&RESPONSE_MAGIC);
    frame.push(PROTOCOL_VERSION);
    frame.push(responder.len() as u8);
    frame.extend_from_slice(responder.as_bytes());
    frame.extend_from_slice(&window_id.to_le_bytes());
    frame.extend_from_slice(&(timesteps.len() as u16).to_le_bytes());
    for timestep in timesteps {
        frame.extend_from_slice(&timestep.to_le_bytes());
    }
    for value in hidden {
        frame.extend_from_slice(&value.to_le_bytes());
    }
    for value in features {
        frame.extend_from_slice(&value.to_le_bytes());
    }
    frame
}

fn decode_request(data: &[u8]) -> Option<Request> {
    let offset = &mut 0;
    if !read_magic(data, offset, &REQUEST_MAGIC) {
        return None;
    }
    let version = read_u8(data, offset)?;
    if version != PROTOCOL_VERSION {
        return None;
    }
    let requester = read_device_id(data, offset)?;
    let window_id = read_u64(data, offset)?;
    let timesteps = read_timesteps(data, offset)?;
    if *offset != data.len() {
        return None;
    }
    Some(Request { requester, window_id, timesteps })
}

fn decode_response(data: &[u8]) -> Option<Response> {
    let state = state()?;
    let offset = &mut 0;
    if !read_magic(data, offset, &RESPONSE_MAGIC) {
        return None;
    }
    let version = read_u8(data, offset)?;
    if version != PROTOCOL_VERSION {
        return None;
    }
    let responder = read_device_id(data, offset)?;
    let window_id = read_u64(data, offset)?;
    let timesteps = read_timesteps(data, offset)?;
    let hidden = read_f32s(data, offset, timesteps.len() * state.hidden_size)?;
    let features = read_f32s(data, offset, timesteps.len() * state.features_per_node)?;
    if *offset != data.len() {
        return None;
    }
    Some(Response {
        responder,
        window_id,
        timesteps,
        hidden,
        features,
    })
}

fn read_magic(data: &[u8], offset: &mut usize, magic: &[u8]) -> bool {
    if *offset + magic.len() > data.len() {
        return false;
    }
    let matched = &data[*offset..*offset + magic.len()] == magic;
    *offset += magic.len();
    matched
}

fn read_u8(data: &[u8], offset: &mut usize) -> Option<u8> {
    if *offset >= data.len() {
        return None;
    }
    let value = data[*offset];
    *offset += 1;
    Some(value)
}

fn read_device_id(data: &[u8], offset: &mut usize) -> Option<String> {
    let length = read_u8(data, offset)? as usize;
    if length == 0 || length > MAX_DEVICE_ID || *offset + length > data.len() {
        return None;
    }
    let id = std::str::from_utf8(&data[*offset..*offset + length]).ok()?.to_owned();
    *offset += length;
    Some(id)
}

fn read_u64(data: &[u8], offset: &mut usize) -> Option<u64> {
    if *offset + 8 > data.len() {
        return None;
    }
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&data[*offset..*offset + 8]);
    *offset += 8;
    Some(u64::from_le_bytes(bytes))
}

fn read_timesteps(data: &[u8], offset: &mut usize) -> Option<Vec<u16>> {
    if *offset + 2 > data.len() {
        return None;
    }
    let count = u16::from_le_bytes([data[*offset], data[*offset + 1]]) as usize;
    *offset += 2;
    if count > MAX_TIMESTEPS {
        return None;
    }
    let mut timesteps = Vec::with_capacity(count);
    for _ in 0..count {
        if *offset + 2 > data.len() {
            return None;
        }
        timesteps.push(u16::from_le_bytes([data[*offset], data[*offset + 1]]));
        *offset += 2;
    }
    Some(timesteps)
}

fn read_f32s(data: &[u8], offset: &mut usize, count: usize) -> Option<Vec<f32>> {
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        if *offset + 4 > data.len() {
            return None;
        }
        let mut bytes = [0u8; 4];
        bytes.copy_from_slice(&data[*offset..*offset + 4]);
        values.push(f32::from_le_bytes(bytes));
        *offset += 4;
    }
    Some(values)
}
