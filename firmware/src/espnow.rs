//! ESP-NOW peer-to-peer transport for the CESTA neighbor exchange.
//!
//! Carries the binary frames from `exchange` directly between nodes over
//! ESP-NOW action frames, so the request/response protocol runs without a
//! broker. ESP-NOW limits a single packet to 250 bytes, while a full-window
//! dense response can reach ~16 KB, so frames are split into fragments:
//!
//! Fragment header (little-endian): magic `CF` u16 | frame_id u16 |
//! fragment index u8 | fragment count u8 | chunk payload
//!
//! The worker thread owns reassembly, request/response draining, and sending;
//! the Wi-Fi task callback only copies raw packets into a queue. Peers are
//! static, taken from `config::NEIGHBORS` MAC addresses, unencrypted, and
//! send on the channel of the station interface (channel 0 = current).

use std::collections::HashMap;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

use esp_idf_svc::sys::{
    ESP_OK, esp_now_add_peer, esp_now_init, esp_now_peer_info_t, esp_now_recv_info_t,
    esp_now_register_recv_cb, esp_now_send, esp_wifi_get_mac, wifi_interface_t_WIFI_IF_STA,
};
use log::{debug, error, info, warn};

use crate::config;
use crate::exchange;

const FRAG_MAGIC: [u8; 2] = *b"CF";
const FRAG_HEADER_BYTES: usize = 6;
const MAX_PACKET_BYTES: usize = 250;
const FRAG_CHUNK_BYTES: usize = MAX_PACKET_BYTES - FRAG_HEADER_BYTES;
/// Largest exchange frame the fragment layer carries; count fits in one byte.
const MAX_FRAME_BYTES: usize = 255 * FRAG_CHUNK_BYTES;
/// Drop incomplete frames after this long; `EXCHANGE_WAIT_MS` bounds interest.
const REASSEMBLY_TIMEOUT: Duration = Duration::from_millis(2_000);
const MAX_PARTIAL_FRAMES: usize = 8;

type Mac = [u8; 6];

/// One outbound exchange frame addressed to a peer MAC.
pub struct EspNowJob {
    pub target: Mac,
    pub payload: Vec<u8>,
}

struct PartialFrame {
    count: u8,
    received: u8,
    chunks: Vec<Option<Vec<u8>>>,
    started: Instant,
}

static RX_QUEUE: Mutex<Option<Sender<(Mac, Vec<u8>)>>> = Mutex::new(None);
static STARTED: OnceLock<()> = OnceLock::new();

/// Start ESP-NOW and the transport worker; returns the outbound job queue.
/// Must be called after Wi-Fi has started (ESP-NOW rides the station
/// interface) and after `exchange::init`.
pub fn start() -> Sender<EspNowJob> {
    let (sender, receiver) = channel();
    if STARTED.set(()).is_err() {
        error!("[ESPNOW] start called twice; second queue is disconnected");
        return sender;
    }
    init_driver();
    let (rx_sender, rx_receiver) = channel();
    {
        let Ok(mut queue) = RX_QUEUE.lock() else {
            panic!("ESP-NOW RX queue mutex poisoned");
        };
        *queue = Some(rx_sender);
    }
    thread::Builder::new()
        .stack_size(16 * 1024)
        .spawn(move || run(receiver, rx_receiver))
        .expect("Failed to spawn ESP-NOW worker");
    sender
}

fn init_driver() {
    let result = unsafe { esp_now_init() };
    if result != ESP_OK {
        panic!("esp_now_init failed: {}", result);
    }
    let result = unsafe { esp_now_register_recv_cb(Some(on_recv)) };
    if result != ESP_OK {
        panic!("esp_now_register_recv_cb failed: {}", result);
    }

    let mut own_mac = [0u8; 6];
    let result = unsafe { esp_wifi_get_mac(wifi_interface_t_WIFI_IF_STA, own_mac.as_mut_ptr()) };
    if result == ESP_OK {
        info!(
            "[ESPNOW] own MAC {:02X}:{:02X}:{:02X}:{:02X}:{:02X}:{:02X}",
            own_mac[0], own_mac[1], own_mac[2], own_mac[3], own_mac[4], own_mac[5]
        );
    }

    for neighbor in config::NEIGHBORS.iter() {
        let mut peer = esp_now_peer_info_t::default();
        peer.peer_addr.copy_from_slice(&neighbor.mac);
        peer.channel = 0; // current channel of the station interface
        peer.ifidx = wifi_interface_t_WIFI_IF_STA;
        peer.encrypt = false;
        let result = unsafe { esp_now_add_peer(&peer) };
        if result != ESP_OK {
            error!(
                "[ESPNOW] failed to add peer {} ({:02X?}): err={}",
                neighbor.device_id, neighbor.mac, result
            );
        } else {
            info!(
                "[ESPNOW] peer added {} ({:02X?})",
                neighbor.device_id, neighbor.mac
            );
        }
    }
}

/// RX callback executed in the Wi-Fi task; only copies the packet and queues
/// it for the worker thread.
unsafe extern "C" fn on_recv(info: *const esp_now_recv_info_t, data: *const u8, data_len: i32) {
    if info.is_null() || data.is_null() || data_len <= 0 {
        return;
    }
    let src = unsafe { (*info).src_addr };
    if src.is_null() {
        return;
    }
    let mut mac = [0u8; 6];
    unsafe { std::ptr::copy_nonoverlapping(src, mac.as_mut_ptr(), 6) };
    let packet = unsafe { std::slice::from_raw_parts(data, data_len as usize) }.to_vec();
    let Ok(queue) = RX_QUEUE.lock() else {
        return;
    };
    if let Some(queue) = queue.as_ref()
        && queue.send((mac, packet)).is_err()
    {
        // Worker is gone; nothing more to do.
    }
}

fn run(jobs: Receiver<EspNowJob>, rx: Receiver<(Mac, Vec<u8>)>) {
    let mut next_frame_id: u16 = 0;
    let mut partials: HashMap<(Mac, u16), PartialFrame> = HashMap::new();
    let peers: HashMap<&str, Mac> = config::NEIGHBORS
        .iter()
        .map(|neighbor| (neighbor.device_id, neighbor.mac))
        .collect();
    loop {
        for (src, packet) in rx.try_iter() {
            if let Some(frame) = reassemble(&mut partials, src, &packet) {
                exchange::handle_frame(&frame);
            }
        }
        partials.retain(|_, partial| partial.started.elapsed() < REASSEMBLY_TIMEOUT);
        for job in jobs.try_iter() {
            next_frame_id = next_frame_id.wrapping_add(1);
            send_frame(&job.target, next_frame_id, &job.payload);
        }
        for (requester, payload) in exchange::serve_pending_requests() {
            next_frame_id = next_frame_id.wrapping_add(1);
            match peers.get(requester.as_str()) {
                Some(mac) => send_frame(mac, next_frame_id, &payload),
                None => warn!(
                    "[ESPNOW] no peer MAC for requester {}, response dropped",
                    requester
                ),
            }
        }
        thread::sleep(Duration::from_millis(config::EXCHANGE_POLL_MS));
    }
}

/// Feed one received fragment into the reassembly table; returns the
/// reassembled exchange frame once all of its fragments have arrived.
fn reassemble(
    partials: &mut HashMap<(Mac, u16), PartialFrame>,
    src: Mac,
    packet: &[u8],
) -> Option<Vec<u8>> {
    if packet.len() < FRAG_HEADER_BYTES || packet[0..2] != FRAG_MAGIC {
        debug!("[ESPNOW] dropped packet without fragment header");
        return None;
    }
    let frame_id = u16::from_le_bytes([packet[2], packet[3]]);
    let index = packet[4];
    let count = packet[5];
    if count == 0 || index >= count {
        warn!(
            "[ESPNOW] dropped fragment with invalid index {}/{}",
            index, count
        );
        return None;
    }
    let chunk = &packet[FRAG_HEADER_BYTES..];
    if index == count - 1 && chunk.is_empty() {
        warn!("[ESPNOW] dropped empty trailing fragment");
        return None;
    }
    if count == 1 {
        return Some(chunk.to_vec());
    }

    let key = (src, frame_id);
    if !partials.contains_key(&key) {
        if partials.len() >= MAX_PARTIAL_FRAMES {
            if let Some(oldest) = partials
                .iter()
                .max_by_key(|(_, partial)| partial.started.elapsed())
                .map(|(key, _)| *key)
            {
                partials.remove(&oldest);
            }
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
    if partial.count != count {
        warn!(
            "[ESPNOW] fragment count mismatch for frame {}, dropped",
            frame_id
        );
        partials.remove(&key);
        return None;
    }
    if partial.chunks[index as usize].is_none() {
        partial.received += 1;
    }
    partial.chunks[index as usize] = Some(chunk.to_vec());
    if partial.received != partial.count {
        return None;
    }

    let partial = partials.remove(&key)?;
    let mut frame = Vec::new();
    for chunk in partial.chunks.into_iter().flatten() {
        frame.extend_from_slice(&chunk);
    }
    debug!(
        "[ESPNOW] reassembled frame {} from {:02X?} ({} bytes)",
        frame_id,
        src,
        frame.len()
    );
    Some(frame)
}

/// Fragment and send one exchange frame to a peer.
fn send_frame(target: &Mac, frame_id: u16, payload: &[u8]) {
    if payload.len() > MAX_FRAME_BYTES {
        error!(
            "[ESPNOW] frame of {} bytes exceeds fragment capacity, dropped",
            payload.len()
        );
        return;
    }
    let count = payload.len().div_ceil(FRAG_CHUNK_BYTES).max(1) as u8;
    for index in 0..count {
        let start = index as usize * FRAG_CHUNK_BYTES;
        let end = (start + FRAG_CHUNK_BYTES).min(payload.len());
        let mut packet = Vec::with_capacity(FRAG_HEADER_BYTES + (end - start));
        packet.extend_from_slice(&FRAG_MAGIC);
        packet.extend_from_slice(&frame_id.to_le_bytes());
        packet.push(index);
        packet.push(count);
        packet.extend_from_slice(&payload[start..end]);
        let result = unsafe { esp_now_send(target.as_ptr(), packet.as_ptr(), packet.len()) };
        if result != ESP_OK {
            error!(
                "[ESPNOW] send to {:02X?} failed at fragment {}/{}: err={}",
                target,
                index + 1,
                count,
                result
            );
            return;
        }
    }
}
