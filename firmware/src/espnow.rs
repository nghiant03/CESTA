use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, TrySendError, sync_channel};
use std::sync::{Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

use esp_idf_hal::delay::FreeRtos;
use esp_idf_svc::sys::{
    ESP_OK, esp_now_add_peer, esp_now_init, esp_now_peer_info_t, esp_now_recv_info_t,
    esp_now_register_recv_cb, esp_now_register_send_cb, esp_now_send, esp_now_send_status_t,
    esp_now_send_status_t_ESP_NOW_SEND_SUCCESS, esp_wifi_get_mac, wifi_interface_t_WIFI_IF_STA,
};
use log::{error, info, warn};

use crate::{config, exchange};

const FRAG_MAGIC: [u8; 2] = *b"CF";
const FRAG_HEADER_BYTES: usize = 6;
const MAX_PACKET_BYTES: usize = 250;
const FRAG_CHUNK_BYTES: usize = MAX_PACKET_BYTES - FRAG_HEADER_BYTES;
pub const MAX_FRAME_BYTES: usize = 255 * FRAG_CHUNK_BYTES;
const MAX_PARTIAL_FRAMES: usize = 2;
const RX_CAPACITY: usize = 64;
const TX_CAPACITY: usize = 4;
const MAX_PEERS: usize = 20;
const REASSEMBLY_TIMEOUT: Duration = Duration::from_secs(2);
const SEND_TIMEOUT: Duration = Duration::from_millis(500);

type Mac = [u8; 6];

pub struct EspNowJob {
    pub target: Mac,
    pub payload: Vec<u8>,
}

#[derive(Clone)]
pub struct EspNowSender(SyncSender<EspNowJob>);

impl EspNowSender {
    pub fn send(&self, job: EspNowJob) -> Result<(), TrySendError<EspNowJob>> {
        if !HEALTHY.load(Ordering::Acquire)
            || job.payload.is_empty()
            || job.payload.len() > MAX_FRAME_BYTES
            || !registered(&job.target)
        {
            TX_ADMISSION_REJECTIONS.add(1);
            return Err(TrySendError::Disconnected(job));
        }
        self.0.try_send(job).inspect_err(|_| TX_QUEUE_DROPS.add(1))
    }
}

struct PartialFrame {
    count: u8,
    received: u8,
    chunks: Vec<Option<Vec<u8>>>,
    started: Instant,
}

#[cfg(target_has_atomic = "64")]
struct Counter(std::sync::atomic::AtomicU64);

#[cfg(not(target_has_atomic = "64"))]
struct Counter {
    section: esp_idf_hal::interrupt::IsrCriticalSection,
    low: std::sync::atomic::AtomicU32,
    high: std::sync::atomic::AtomicU32,
}

impl Counter {
    const fn new() -> Self {
        #[cfg(target_has_atomic = "64")]
        {
            Self(std::sync::atomic::AtomicU64::new(0))
        }
        #[cfg(not(target_has_atomic = "64"))]
        {
            Self {
                section: esp_idf_hal::interrupt::IsrCriticalSection::new(),
                low: std::sync::atomic::AtomicU32::new(0),
                high: std::sync::atomic::AtomicU32::new(0),
            }
        }
    }

    fn add(&self, value: u64) {
        #[cfg(target_has_atomic = "64")]
        {
            self.0.fetch_add(value, Ordering::Relaxed);
        }
        #[cfg(not(target_has_atomic = "64"))]
        {
            let _guard = self.section.enter();
            let count = (u64::from(self.high.load(Ordering::Relaxed)) << 32)
                | u64::from(self.low.load(Ordering::Relaxed));
            let next = count.wrapping_add(value);
            self.low.store(next as u32, Ordering::Relaxed);
            self.high.store((next >> 32) as u32, Ordering::Relaxed);
        }
    }

    fn load(&self) -> u64 {
        #[cfg(target_has_atomic = "64")]
        {
            self.0.load(Ordering::Relaxed)
        }
        #[cfg(not(target_has_atomic = "64"))]
        {
            let _guard = self.section.enter();
            (u64::from(self.high.load(Ordering::Relaxed)) << 32)
                | u64::from(self.low.load(Ordering::Relaxed))
        }
    }
}

static TX_FRAMES_ATTEMPTED: Counter = Counter::new();
static TX_FRAMES_COMPLETED: Counter = Counter::new();
static TX_FRAMES_FAILED: Counter = Counter::new();
static TX_FRAGMENTS_ATTEMPTED: Counter = Counter::new();
static TX_FRAGMENTS_ACCEPTED: Counter = Counter::new();
static TX_FRAGMENTS_COMPLETED: Counter = Counter::new();
static TX_FRAGMENTS_FAILED: Counter = Counter::new();
static TX_FRAGMENTS_REJECTED: Counter = Counter::new();
static TX_FRAGMENTS_DELIVERY_FAILED: Counter = Counter::new();
static TX_FRAGMENTS_OUTCOME_UNKNOWN: Counter = Counter::new();
static TX_FRAGMENT_BYTES_ATTEMPTED: Counter = Counter::new();
static TX_FRAGMENT_BYTES_ACCEPTED: Counter = Counter::new();
static TX_FRAGMENT_BYTES_COMPLETED: Counter = Counter::new();
static TX_QUEUE_DROPS: Counter = Counter::new();
static TX_ADMISSION_REJECTIONS: Counter = Counter::new();
static TX_RESPONSE_DROPS: Counter = Counter::new();
static TX_COMPLETION_QUEUE_DROPS: Counter = Counter::new();
static RX_FRAGMENTS: Counter = Counter::new();
static RX_FRAGMENT_BYTES: Counter = Counter::new();
static RX_FRAMES_REASSEMBLED: Counter = Counter::new();
static RX_QUEUE_DROPS: Counter = Counter::new();
static RX_CALLBACK_INVALID: Counter = Counter::new();
static RX_MALFORMED_DROPS: Counter = Counter::new();
static RX_REASSEMBLY_EXPIRED: Counter = Counter::new();
static RX_REASSEMBLY_CONFLICTS: Counter = Counter::new();
static RX_REASSEMBLY_CAPACITY_DROPS: Counter = Counter::new();
static RX_DUPLICATE_FRAGMENTS: Counter = Counter::new();
static RX_ORIGIN_DROPS: Counter = Counter::new();

pub fn telemetry_snapshot() -> serde_json::Value {
    serde_json::json!({
        "tx_frames_attempted": TX_FRAMES_ATTEMPTED.load(),
        "tx_frames_completed": TX_FRAMES_COMPLETED.load(),
        "tx_frames_failed": TX_FRAMES_FAILED.load(),
        "tx_fragments_attempted": TX_FRAGMENTS_ATTEMPTED.load(),
        "tx_fragments_accepted": TX_FRAGMENTS_ACCEPTED.load(),
        "tx_fragments_completed": TX_FRAGMENTS_COMPLETED.load(),
        "tx_fragments_failed": TX_FRAGMENTS_FAILED.load(),
        "tx_fragments_rejected": TX_FRAGMENTS_REJECTED.load(),
        "tx_fragments_delivery_failed": TX_FRAGMENTS_DELIVERY_FAILED.load(),
        "tx_fragments_outcome_unknown": TX_FRAGMENTS_OUTCOME_UNKNOWN.load(),
        "tx_fragment_bytes_attempted": TX_FRAGMENT_BYTES_ATTEMPTED.load(),
        "tx_fragment_bytes_accepted": TX_FRAGMENT_BYTES_ACCEPTED.load(),
        "tx_fragment_bytes_completed": TX_FRAGMENT_BYTES_COMPLETED.load(),
        "tx_queue_drops": TX_QUEUE_DROPS.load(),
        "tx_admission_rejections": TX_ADMISSION_REJECTIONS.load(),
        "tx_response_drops": TX_RESPONSE_DROPS.load(),
        "tx_completion_queue_drops": TX_COMPLETION_QUEUE_DROPS.load(),
        "rx_fragments": RX_FRAGMENTS.load(),
        "rx_fragment_bytes": RX_FRAGMENT_BYTES.load(),
        "rx_frames_reassembled": RX_FRAMES_REASSEMBLED.load(),
        "rx_queue_drops": RX_QUEUE_DROPS.load(),
        "rx_callback_invalid": RX_CALLBACK_INVALID.load(),
        "rx_malformed_drops": RX_MALFORMED_DROPS.load(),
        "rx_reassembly_expired": RX_REASSEMBLY_EXPIRED.load(),
        "rx_reassembly_conflicts": RX_REASSEMBLY_CONFLICTS.load(),
        "rx_reassembly_capacity_drops": RX_REASSEMBLY_CAPACITY_DROPS.load(),
        "rx_duplicate_fragments": RX_DUPLICATE_FRAGMENTS.load(),
        "rx_origin_drops": RX_ORIGIN_DROPS.load()
    })
}

static RX_QUEUE: Mutex<Option<SyncSender<(Mac, Vec<u8>)>>> = Mutex::new(None);
static COMPLETIONS: Mutex<Option<SyncSender<(Mac, bool)>>> = Mutex::new(None);
static PEERS: Mutex<Vec<Mac>> = Mutex::new(Vec::new());
static STARTED: OnceLock<()> = OnceLock::new();
static HEALTHY: AtomicBool = AtomicBool::new(false);

fn valid_mac(mac: &Mac) -> bool {
    *mac != [0; 6] && mac[0] & 1 == 0
}

fn configured_valid(index: usize, own_mac: &Mac) -> bool {
    let neighbor = &config::NEIGHBORS[index];
    valid_mac(&neighbor.mac)
        && neighbor.mac != *own_mac
        && !neighbor.device_id.is_empty()
        && neighbor.device_id.len() <= 32
        && config::NEIGHBORS.iter().enumerate().all(|(other, peer)| {
            other == index || (peer.mac != neighbor.mac && peer.device_id != neighbor.device_id)
        })
}

fn registered(mac: &Mac) -> bool {
    PEERS.lock().is_ok_and(|peers| peers.contains(mac))
}

pub fn registered_peer_mask() -> Vec<bool> {
    config::NEIGHBORS
        .iter()
        .map(|peer| HEALTHY.load(Ordering::Acquire) && registered(&peer.mac))
        .collect()
}

fn add_peer(mac: Mac) -> bool {
    let Ok(mut peers) = PEERS.lock() else {
        return false;
    };
    if peers.contains(&mac) {
        return true;
    }
    if !valid_mac(&mac) || peers.len() >= MAX_PEERS {
        return false;
    }
    let mut peer = esp_now_peer_info_t::default();
    peer.peer_addr = mac;
    peer.channel = 0;
    peer.ifidx = wifi_interface_t_WIFI_IF_STA;
    peer.encrypt = false;
    let result = unsafe { esp_now_add_peer(&peer) };
    if result != ESP_OK {
        error!("[ESPNOW] registration failed for {:02X?}: {}", mac, result);
        return false;
    }
    peers.push(mac);
    true
}

pub fn start() -> EspNowSender {
    assert!(STARTED.set(()).is_ok(), "ESP-NOW already started");
    let (sender, jobs) = sync_channel(TX_CAPACITY);
    let (rx_sender, rx) = sync_channel(RX_CAPACITY);
    let (completion_sender, completions) = sync_channel(1);
    *RX_QUEUE.lock().expect("RX queue poisoned") = Some(rx_sender);
    *COMPLETIONS.lock().expect("completion queue poisoned") = Some(completion_sender);
    let result = unsafe { esp_now_init() };
    assert_eq!(result, ESP_OK, "esp_now_init failed");
    let mut own_mac = [0; 6];
    let result = unsafe { esp_wifi_get_mac(wifi_interface_t_WIFI_IF_STA, own_mac.as_mut_ptr()) };
    assert_eq!(result, ESP_OK, "cannot read local MAC");
    info!("[ESPNOW] own MAC {:02X?}", own_mac);
    for (index, neighbor) in config::NEIGHBORS.iter().enumerate() {
        if !configured_valid(index, &own_mac) {
            error!(
                "[ESPNOW] invalid or ambiguous peer {}: {:02X?}",
                neighbor.device_id, neighbor.mac
            );
            continue;
        }
        if add_peer(neighbor.mac) {
            info!(
                "[ESPNOW] locally registered {} (RF reachability unknown)",
                neighbor.device_id
            );
        }
    }
    let result = unsafe { esp_now_register_recv_cb(Some(on_recv)) };
    assert_eq!(result, ESP_OK, "receive callback registration failed");
    let result = unsafe { esp_now_register_send_cb(Some(on_send)) };
    assert_eq!(result, ESP_OK, "send callback registration failed");
    HEALTHY.store(true, Ordering::Release);
    thread::Builder::new()
        .stack_size(16 * 1024)
        .spawn(move || run(jobs, rx, completions, own_mac))
        .expect("Failed to spawn ESP-NOW worker");
    EspNowSender(sender)
}

unsafe extern "C" fn on_recv(info: *const esp_now_recv_info_t, data: *const u8, data_len: i32) {
    if info.is_null() || data.is_null() || !(7..=MAX_PACKET_BYTES as i32).contains(&data_len) {
        RX_CALLBACK_INVALID.add(1);
        return;
    }
    let src = unsafe { (*info).src_addr };
    if src.is_null() {
        RX_CALLBACK_INVALID.add(1);
        return;
    }
    let mut mac = [0; 6];
    unsafe { std::ptr::copy_nonoverlapping(src, mac.as_mut_ptr(), 6) };
    if !valid_mac(&mac) {
        RX_CALLBACK_INVALID.add(1);
        return;
    }
    RX_FRAGMENTS.add(1);
    RX_FRAGMENT_BYTES.add(data_len as u64);
    let Ok(queue) = RX_QUEUE.try_lock() else {
        RX_QUEUE_DROPS.add(1);
        return;
    };
    if let Some(queue) = queue.as_ref() {
        let packet = unsafe { std::slice::from_raw_parts(data, data_len as usize) }.to_vec();
        if queue.try_send((mac, packet)).is_err() {
            RX_QUEUE_DROPS.add(1);
        }
    } else {
        RX_QUEUE_DROPS.add(1);
    }
}

unsafe extern "C" fn on_send(mac: *const u8, status: esp_now_send_status_t) {
    if mac.is_null() {
        return;
    }
    let mut target = [0; 6];
    unsafe { std::ptr::copy_nonoverlapping(mac, target.as_mut_ptr(), 6) };
    if let Ok(queue) = COMPLETIONS.try_lock()
        && let Some(queue) = queue.as_ref()
    {
        if queue
            .try_send((target, status == esp_now_send_status_t_ESP_NOW_SEND_SUCCESS))
            .is_err()
        {
            TX_COMPLETION_QUEUE_DROPS.add(1);
        }
    } else {
        TX_COMPLETION_QUEUE_DROPS.add(1);
    }
}

fn frame_identity(frame: &[u8]) -> Option<(&str, bool)> {
    let request = frame.starts_with(b"CESTR");
    if !request && !frame.starts_with(b"CESTP") {
        return None;
    }
    let length = *frame.get(6)? as usize;
    if !(1..=32).contains(&length) {
        return None;
    }
    Some((
        std::str::from_utf8(frame.get(7..7 + length)?).ok()?,
        request,
    ))
}

fn accept_origin(frame: &[u8], src: Mac, own_mac: Mac) -> bool {
    let Some((identity, request)) = frame_identity(frame) else {
        return false;
    };
    let valid = if request {
        exchange::valid_request_frame(frame)
    } else {
        exchange::valid_response_frame(frame)
    };
    if !valid || src == own_mac || identity == config::DEVICE_ID {
        return false;
    }
    config::NEIGHBORS.iter().enumerate().any(|(index, peer)| {
        peer.device_id == identity
            && peer.mac == src
            && configured_valid(index, &own_mac)
            && registered(&src)
    })
}

fn run(
    jobs: Receiver<EspNowJob>,
    rx: Receiver<(Mac, Vec<u8>)>,
    completions: Receiver<(Mac, bool)>,
    own_mac: Mac,
) {
    let mut next_frame_id: u16 = 0;
    let mut partials = HashMap::new();
    loop {
        partials.retain(|_, partial: &mut PartialFrame| {
            let keep = partial.started.elapsed() < REASSEMBLY_TIMEOUT;
            if !keep {
                RX_REASSEMBLY_EXPIRED.add(1);
            }
            keep
        });
        for (src, packet) in rx.try_iter().take(RX_CAPACITY) {
            if let Some(frame) = reassemble(&mut partials, src, &packet) {
                if accept_origin(&frame, src, own_mac) {
                    exchange::handle_frame(&frame);
                } else {
                    RX_ORIGIN_DROPS.add(1);
                }
            }
        }
        if let Ok(job) = jobs.try_recv() {
            next_frame_id = next_frame_id.wrapping_add(1);
            send_frame(&job.target, next_frame_id, &job.payload, &completions);
        }
        for (requester, payload) in exchange::serve_pending_requests() {
            if let Some((_, peer)) = config::NEIGHBORS.iter().enumerate().find(|(index, peer)| {
                peer.device_id == requester
                    && configured_valid(*index, &own_mac)
                    && registered(&peer.mac)
            }) {
                next_frame_id = next_frame_id.wrapping_add(1);
                send_frame(&peer.mac, next_frame_id, &payload, &completions);
            } else {
                TX_RESPONSE_DROPS.add(1);
                warn!(
                    "[ESPNOW] no validated origin for {}, response dropped",
                    requester
                );
            }
        }
        FreeRtos::delay_ms(1);
    }
}

fn reassemble(
    partials: &mut HashMap<(Mac, u16), PartialFrame>,
    src: Mac,
    packet: &[u8],
) -> Option<Vec<u8>> {
    if !(FRAG_HEADER_BYTES + 1..=MAX_PACKET_BYTES).contains(&packet.len())
        || packet[0..2] != FRAG_MAGIC
    {
        RX_MALFORMED_DROPS.add(1);
        return None;
    }
    let frame_id = u16::from_le_bytes([packet[2], packet[3]]);
    let index = packet[4];
    let count = packet[5];
    let chunk = &packet[FRAG_HEADER_BYTES..];
    if count == 0 || index >= count || (index + 1 < count && chunk.len() != FRAG_CHUNK_BYTES) {
        RX_MALFORMED_DROPS.add(1);
        return None;
    }
    let key = (src, frame_id);
    if let Some(partial) = partials.get(&key)
        && (partial.count != count || partial.started.elapsed() >= REASSEMBLY_TIMEOUT)
    {
        if partial.started.elapsed() >= REASSEMBLY_TIMEOUT {
            RX_REASSEMBLY_EXPIRED.add(1);
        } else {
            RX_REASSEMBLY_CONFLICTS.add(1);
        }
        partials.remove(&key);
        return None;
    }
    if count == 1 {
        RX_FRAMES_REASSEMBLED.add(1);
        return Some(chunk.to_vec());
    }
    if !partials.contains_key(&key) {
        if partials.len() >= MAX_PARTIAL_FRAMES {
            RX_REASSEMBLY_CAPACITY_DROPS.add(1);
            return None;
        }
        partials.insert(
            key,
            PartialFrame {
                count,
                received: 0,
                chunks: vec![None; count as usize],
                started: Instant::now(),
            },
        );
    }
    let partial = partials.get_mut(&key)?;
    if let Some(previous) = &partial.chunks[index as usize] {
        if previous != chunk {
            RX_REASSEMBLY_CONFLICTS.add(1);
            partials.remove(&key);
        } else {
            RX_DUPLICATE_FRAGMENTS.add(1);
        }
        return None;
    }
    partial.chunks[index as usize] = Some(chunk.to_vec());
    partial.received += 1;
    if partial.received != partial.count {
        return None;
    }
    let partial = partials.remove(&key)?;
    let size = partial.chunks.iter().flatten().map(Vec::len).sum();
    let mut frame = Vec::with_capacity(size);
    for chunk in partial.chunks.into_iter().flatten() {
        frame.extend_from_slice(&chunk);
    }
    RX_FRAMES_REASSEMBLED.add(1);
    Some(frame)
}

fn send_frame(
    target: &Mac,
    frame_id: u16,
    payload: &[u8],
    completions: &Receiver<(Mac, bool)>,
) -> bool {
    TX_FRAMES_ATTEMPTED.add(1);
    let completed = send_frame_inner(target, frame_id, payload, completions);
    if completed {
        TX_FRAMES_COMPLETED.add(1);
    } else {
        TX_FRAMES_FAILED.add(1);
    }
    completed
}

fn send_frame_inner(
    target: &Mac,
    frame_id: u16,
    payload: &[u8],
    completions: &Receiver<(Mac, bool)>,
) -> bool {
    if payload.is_empty()
        || payload.len() > MAX_FRAME_BYTES
        || !registered(target)
        || !HEALTHY.load(Ordering::Acquire)
    {
        return false;
    }
    let count = payload.len().div_ceil(FRAG_CHUNK_BYTES) as u8;
    for (index, chunk) in payload.chunks(FRAG_CHUNK_BYTES).enumerate() {
        let mut packet = [0; MAX_PACKET_BYTES];
        packet[..2].copy_from_slice(&FRAG_MAGIC);
        packet[2..4].copy_from_slice(&frame_id.to_le_bytes());
        packet[4] = index as u8;
        packet[5] = count;
        packet[FRAG_HEADER_BYTES..FRAG_HEADER_BYTES + chunk.len()].copy_from_slice(chunk);
        let bytes = (FRAG_HEADER_BYTES + chunk.len()) as u64;
        TX_FRAGMENTS_ATTEMPTED.add(1);
        TX_FRAGMENT_BYTES_ATTEMPTED.add(bytes);
        let result = unsafe {
            esp_now_send(
                target.as_ptr(),
                packet.as_ptr(),
                FRAG_HEADER_BYTES + chunk.len(),
            )
        };
        if result != ESP_OK {
            TX_FRAGMENTS_FAILED.add(1);
            TX_FRAGMENTS_REJECTED.add(1);
            warn!("[ESPNOW] send rejected for {:02X?}: {}", target, result);
            return false;
        }
        TX_FRAGMENTS_ACCEPTED.add(1);
        TX_FRAGMENT_BYTES_ACCEPTED.add(bytes);
        match completions.recv_timeout(SEND_TIMEOUT) {
            Ok((mac, true)) if mac == *target => {
                TX_FRAGMENTS_COMPLETED.add(1);
                TX_FRAGMENT_BYTES_COMPLETED.add(bytes);
            }
            Ok((mac, false)) if mac == *target => {
                TX_FRAGMENTS_FAILED.add(1);
                TX_FRAGMENTS_DELIVERY_FAILED.add(1);
                warn!("[ESPNOW] delivery failed for {:02X?}", target);
                return false;
            }
            _ => {
                TX_FRAGMENTS_FAILED.add(1);
                TX_FRAGMENTS_OUTCOME_UNKNOWN.add(1);
                HEALTHY.store(false, Ordering::Release);
                error!("[ESPNOW] completion lost or mismatched; TX disabled until restart");
                return false;
            }
        }
        FreeRtos::delay_ms(1);
    }
    true
}
