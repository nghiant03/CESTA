use std::collections::VecDeque;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use log::{debug, info, warn};

use crate::config;

const REQUEST_MAGIC: [u8; 5] = *b"CESTR";
const RESPONSE_MAGIC: [u8; 5] = *b"CESTP";
const PROTOCOL_VERSION: u8 = 5;
const SAMPLE_INTERVAL_SECS: u64 = 60;
const MAX_INBOX_REQUESTS: usize = 8;
const MAX_INBOX_RESPONSES: usize = 8;
const MAX_DEVICE_ID: usize = 32;
const MAX_TIMESTEPS: usize = 60;

struct SharedState {
    model_fingerprint: [u8; 32],
    window_size: usize,
    hidden_size: usize,
    cache: Mutex<HiddenCache>,
    inbox: Mutex<Inbox>,
}

struct HiddenCache {
    window_id: u64,
    ready: bool,
    deadline: Option<Instant>,
    rounds: [Option<Vec<f32>>; 2],
}

struct Inbox {
    active_response: Option<(u64, u64)>,
    requests: VecDeque<PendingRequest>,
    responses: Vec<Response>,
}

struct PendingRequest {
    request: Request,
    deadline: Instant,
}

static STATE: OnceLock<SharedState> = OnceLock::new();

#[derive(Clone)]
pub struct Request {
    pub requester: String,
    pub window_id: u64,
    pub request_nonce: u64,
    pub timesteps: Vec<u16>,
}

#[derive(Clone)]
pub struct Response {
    pub responder: String,
    pub window_id: u64,
    pub request_nonce: u64,
    pub timesteps: Vec<u16>,
    pub hidden: Vec<f32>,
}

pub fn init(
    window_size: usize,
    hidden_size: usize,
    _features_per_node: usize,
    model_fingerprint: &str,
) -> Result<(), String> {
    if model_fingerprint.len() != 64 || !model_fingerprint.is_ascii() {
        return Err("model_fingerprint must be a SHA256 hex digest".to_owned());
    }
    let mut fingerprint = [0u8; 32];
    for (index, byte) in fingerprint.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&model_fingerprint[index * 2..index * 2 + 2], 16)
            .map_err(|_| "model_fingerprint must be a SHA256 hex digest".to_owned())?;
    }
    if window_size != MAX_TIMESTEPS || hidden_size == 0 || hidden_size > u16::MAX as usize {
        return Err("unsupported exchange dimensions".to_owned());
    }
    STATE
        .set(SharedState {
            model_fingerprint: fingerprint,
            window_size,
            hidden_size,
            cache: Mutex::new(HiddenCache {
                rounds: [None, None],
                window_id: 0,
                ready: false,
                deadline: None,
            }),
            inbox: Mutex::new(Inbox {
                active_response: None,
                requests: VecDeque::new(),
                responses: Vec::new(),
            }),
        })
        .map_err(|_| "exchange already initialized".to_owned())
}

fn state() -> Option<&'static SharedState> {
    STATE.get()
}

pub fn handle_frame(data: &[u8]) {
    let Some(deadline) = transport_deadline().filter(|deadline| Instant::now() < *deadline) else {
        return;
    };
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
            inbox
                .requests
                .retain(|pending| Instant::now() < pending.deadline);
            let duplicate = inbox.requests.iter().any(|pending| {
                pending.request.requester == request.requester
                    && pending.request.window_id == request.window_id
                    && pending.request.request_nonce == request.request_nonce
            });
            if !duplicate && inbox.requests.len() < MAX_INBOX_REQUESTS {
                inbox.requests.push_back(PendingRequest {
                    request,
                    deadline: deadline
                        .min(Instant::now() + Duration::from_millis(config::EXCHANGE_CUTOFF_MS)),
                });
            }
        }
    } else if let Some(response) = decode_response(data) {
        debug!(
            "[EXCHANGE] response from {} for window {} on {} timesteps",
            response.responder,
            response.window_id,
            response.timesteps.len()
        );
        if let Some(state) = state()
            && let Ok(mut inbox) = state.inbox.lock()
        {
            if inbox.active_response == Some((response.window_id, response.request_nonce))
                && inbox.responses.len() < MAX_INBOX_RESPONSES
            {
                inbox.responses.push(response);
            }
        }
    } else {
        warn!("[EXCHANGE] dropped malformed exchange frame");
    }
}

pub fn invalidate_cache() {
    if let Some(state) = state() {
        if let Ok(mut cache) = state.cache.lock() {
            cache.ready = false;
            cache.rounds = [None, None];
            cache.deadline = None;
        }
        if let Ok(mut inbox) = state.inbox.lock() {
            inbox.requests.clear();
            inbox.responses.clear();
            inbox.active_response = None;
        }
    }
}

pub fn prepare_window(window_id: u64, deadline: Instant) {
    if let Some(state) = state()
        && let Ok(mut cache) = state.cache.lock()
    {
        cache.rounds = [None, None];
        cache.ready = false;
        cache.window_id = window_id;
        cache.deadline = Some(deadline);
    }
}

pub fn transport_deadline() -> Option<Instant> {
    state()?.cache.lock().ok()?.deadline
}

pub fn update_round(window_id: u64, round: usize, hidden: &[f32]) -> bool {
    let Some(state) = state() else {
        return false;
    };
    let Ok(mut cache) = state.cache.lock() else {
        return false;
    };
    if round >= 2
        || cache.window_id != window_id
        || !valid_window(window_id)
        || cache.deadline.is_none_or(|d| Instant::now() >= d)
        || hidden.len() != MAX_TIMESTEPS * super::inference::WIDTHS[round]
        || hidden.iter().any(|v| !v.is_finite())
    {
        return false;
    }
    cache.rounds[round] = Some(hidden.to_vec());
    cache.ready = true;
    true
}

fn round(nonce: u64) -> Option<usize> {
    let round = (nonce >> 56) as usize;
    (round < 2).then_some(round)
}

fn valid_window(end: u64) -> bool {
    end >= 1_700_000_000 && end % SAMPLE_INTERVAL_SECS == 0
}

fn window_start(end: u64) -> u64 {
    end.saturating_sub((MAX_TIMESTEPS as u64 - 1) * SAMPLE_INTERVAL_SECS)
}

pub fn serve_pending_requests() -> Vec<(String, Vec<u8>, Instant)> {
    let Some(state) = state() else {
        return Vec::new();
    };
    let Ok(cache) = state.cache.lock() else {
        return Vec::new();
    };
    let Some(deadline) = cache.deadline.filter(|deadline| Instant::now() < *deadline) else {
        return Vec::new();
    };
    let requests = {
        let Ok(mut inbox) = state.inbox.lock() else {
            return Vec::new();
        };
        let mut ready = Vec::new();
        inbox.requests.retain(|pending| {
            let request = &pending.request;
            if Instant::now() >= pending.deadline
                || request.window_id < cache.window_id
                || request.timesteps.is_empty()
            {
                return false;
            }
            if cache.rounds[round(request.request_nonce).unwrap()].is_some()
                && request.window_id == cache.window_id
            {
                ready.push(request.clone());
                return false;
            }
            true
        });
        ready
    };
    requests
        .into_iter()
        .filter_map(|request| {
            if !cache.ready || request.window_id != cache.window_id || request.timesteps.is_empty()
            {
                return None;
            }
            let r = round(request.request_nonce)?;
            let width = super::inference::WIDTHS[r];
            let source = cache.rounds[r].as_ref()?;
            let mut hidden = Vec::with_capacity(request.timesteps.len() * width);
            for timestep in &request.timesteps {
                let timestep = *timestep as usize;
                hidden.extend_from_slice(&source[timestep * width..(timestep + 1) * width]);
            }
            Some((
                request.requester,
                encode_response(
                    config::DEVICE_ID,
                    cache.window_id,
                    request.request_nonce,
                    &request.timesteps,
                    &hidden,
                ),
                deadline,
            ))
        })
        .collect()
}

pub fn begin_response_cycle(window_id: u64, request_nonce: u64) {
    if let Some(state) = state()
        && let Ok(mut inbox) = state.inbox.lock()
    {
        inbox.responses.clear();
        inbox.active_response = Some((window_id, request_nonce));
    }
}

pub fn end_response_cycle() {
    if let Some(state) = state()
        && let Ok(mut inbox) = state.inbox.lock()
    {
        inbox.active_response = None;
        inbox.responses.clear();
    }
}

pub fn take_responses() -> Vec<Response> {
    let Some(state) = state() else {
        return Vec::new();
    };
    let Ok(mut inbox) = state.inbox.lock() else {
        return Vec::new();
    };
    std::mem::take(&mut inbox.responses)
}

pub fn encode_request(
    requester: &str,
    window_id: u64,
    request_nonce: u64,
    timesteps: &[u16],
) -> Vec<u8> {
    let mut frame =
        Vec::with_capacity(REQUEST_MAGIC.len() + 11 + requester.len() + timesteps.len() * 2);
    frame.extend_from_slice(&REQUEST_MAGIC);
    frame.push(PROTOCOL_VERSION);
    frame.push(requester.len() as u8);
    frame.extend_from_slice(requester.as_bytes());
    if !write_compatibility(&mut frame) {
        return Vec::new();
    }
    frame.extend_from_slice(&window_start(window_id).to_le_bytes());
    frame.extend_from_slice(&window_id.to_le_bytes());
    frame.extend_from_slice(&request_nonce.to_le_bytes());
    frame.extend_from_slice(&(timesteps.len() as u16).to_le_bytes());
    for timestep in timesteps {
        frame.extend_from_slice(&timestep.to_le_bytes());
    }
    frame
}

pub fn encode_response(
    responder: &str,
    window_id: u64,
    request_nonce: u64,
    timesteps: &[u16],
    hidden: &[f32],
) -> Vec<u8> {
    let mut frame = Vec::with_capacity(
        RESPONSE_MAGIC.len() + 11 + responder.len() + timesteps.len() * 6 + hidden.len() * 4,
    );
    frame.extend_from_slice(&RESPONSE_MAGIC);
    frame.push(PROTOCOL_VERSION);
    frame.push(responder.len() as u8);
    frame.extend_from_slice(responder.as_bytes());
    if !write_compatibility(&mut frame) {
        return Vec::new();
    }
    frame.extend_from_slice(&window_start(window_id).to_le_bytes());
    frame.extend_from_slice(&window_id.to_le_bytes());
    frame.extend_from_slice(&request_nonce.to_le_bytes());
    frame.extend_from_slice(&(timesteps.len() as u16).to_le_bytes());
    for timestep in timesteps {
        frame.extend_from_slice(&timestep.to_le_bytes());
    }
    for value in hidden {
        frame.extend_from_slice(&value.to_le_bytes());
    }
    frame
}

fn write_compatibility(frame: &mut Vec<u8>) -> bool {
    let Some(state) = state() else {
        return false;
    };
    frame.extend_from_slice(&state.model_fingerprint);
    for value in [
        state.window_size,
        state.hidden_size,
        0,
        SAMPLE_INTERVAL_SECS as usize,
    ] {
        frame.extend_from_slice(&(value as u16).to_le_bytes());
    }
    true
}

fn read_compatibility(data: &[u8], offset: &mut usize) -> bool {
    let mut expected = Vec::with_capacity(40);
    if !write_compatibility(&mut expected) {
        return false;
    }
    if data.get(*offset..*offset + expected.len()) != Some(expected.as_slice()) {
        return false;
    }
    *offset += expected.len();
    true
}

pub fn valid_request_frame(frame: &[u8]) -> bool {
    decode_request(frame).is_some()
}

pub fn valid_response_frame(frame: &[u8]) -> bool {
    decode_response(frame).is_some()
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
    if !read_compatibility(data, offset) {
        return None;
    }
    let start = read_u64(data, offset)?;
    let window_id = read_u64(data, offset)?;
    if !valid_window(window_id) || start != window_start(window_id) {
        return None;
    }
    let request_nonce = read_u64(data, offset)?;
    round(request_nonce)?;
    let timesteps = read_timesteps(data, offset)?;
    if *offset != data.len() {
        return None;
    }
    Some(Request {
        requester,
        window_id,
        request_nonce,
        timesteps,
    })
}

fn decode_response(data: &[u8]) -> Option<Response> {
    state()?;
    let offset = &mut 0;
    if !read_magic(data, offset, &RESPONSE_MAGIC) {
        return None;
    }
    let version = read_u8(data, offset)?;
    if version != PROTOCOL_VERSION {
        return None;
    }
    let responder = read_device_id(data, offset)?;
    if !read_compatibility(data, offset) {
        return None;
    }
    let start = read_u64(data, offset)?;
    let window_id = read_u64(data, offset)?;
    if !valid_window(window_id) || start != window_start(window_id) {
        return None;
    }
    let request_nonce = read_u64(data, offset)?;
    let r = round(request_nonce)?;
    let timesteps = read_timesteps(data, offset)?;
    let hidden = read_f32s(data, offset, timesteps.len() * super::inference::WIDTHS[r])?;
    if *offset != data.len() {
        return None;
    }
    Some(Response {
        responder,
        window_id,
        request_nonce,
        timesteps,
        hidden,
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
    let id = std::str::from_utf8(&data[*offset..*offset + length])
        .ok()?
        .to_owned();
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
        let timestep = u16::from_le_bytes([data[*offset], data[*offset + 1]]);
        if timestep as usize >= state()?.window_size || timesteps.contains(&timestep) {
            return None;
        }
        timesteps.push(timestep);
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
        let value = f32::from_le_bytes(bytes);
        if !value.is_finite() {
            return None;
        }
        values.push(value);
        *offset += 4;
    }
    Some(values)
}
