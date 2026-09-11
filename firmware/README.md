# CESTA firmware

ESP32-S3 firmware that turns each board into a distributed CESTA node. Every node samples a DHT11 sensor, encodes its local 60-sample window with an exported TensorFlow Lite Micro model, requests selected hidden-state payloads from its neighbors directly over ESP-NOW (no broker), aggregates the replies, and publishes readings and per-timestep `NORMAL`/`SPIKE`/`DRIFT`/`STUCK` diagnoses over MQTT.

## Hardware and toolchain requirements

- ESP32-S3 board with **octal PSRAM** and at least **8 MB flash**. The tensor arena and large window buffers are allocated in PSRAM; `sdkconfig.defaults` already carries the required PSRAM and flash settings.
- Host tools: [`espup`](https://github.com/esp-rs/espup), [`espflash`](https://github.com/esp-rs/espflash), `ldproxy`, and the ESP-IDF host dependencies (`rust-toolchain.toml` pins the `esp` channel).

## Configuration

One firmware image is built per deployed node. Edit `src/config.rs`:

| Setting | Purpose |
|---|---|
| `WIFI_SSID` / `WIFI_PASSWORD` | Station credentials (ESP-NOW rides the station interface) |
| `MQTT_SERVER` / `MQTT_PORT` / `MQTT_USER` / `MQTT_PASSWORD` | Telemetry broker |
| `DEVICE_ID` | Unique node name; used in topics and exchange frames |
| `NODE_INDEX` | Graph node index; must match the export's `--receiver-index` |
| `NEIGHBORS` | Graph senders in `sender_indices` order: `device_id`, `node_index`, and station MAC (every node logs its own MAC at boot) |
| `DHT_PIN` / `SPIKE_DHT_PIN` | Sensor pins (normal path / SPIKE fault path) |
| `INFERENCE_ENABLED` / `INFERENCE_TENSOR_ARENA_BYTES` | Toggle and size the TFLite Micro arena |
| `NTP_SERVER` and timing | SNTP clock sync for timestamps |
| `EXCHANGE_WAIT_MS` / `EXCHANGE_POLL_MS` | Diagnosis-cycle exchange deadline and worker poll interval |
| `FAULT_CONFIG` | Fault profile (see below) |

## Fault profiles

`FAULT_CONFIG` supports one profile per fault type in the main codebase:

- `FAULT_NORMAL` — unmodified readings.
- `FAULT_SPIKE` — **hardware** profile. Reads `SPIKE_DHT_PIN` with checksum validation disabled and expects a MOSFET or open-drain transistor to disturb that sensor's DATA line.
- `FAULT_DRIFT` — **software-injected** on the normal reading: linear drift of `drift_rate` °C per sample with a random direction per event, over consecutive events of `event_duration_samples` samples. Mirrors `DriftFaultInjector` in the Python codebase.
- `FAULT_STUCK` — **software-injected** on the normal reading: freezes at the first reading of each event with Gaussian jitter of `jitter_std` °C. Mirrors `StuckFaultInjector`.

Faulted readings are published as fault-path telemetry (`"path": "fault"`); the normal-path reading always feeds inference.

## Model export

Export a trained checkpoint before flashing. The checked-in `model/` artifact only validates conversion and is rejected by the firmware until replaced with a trained export.

```bash
# From the repository root
uv run --isolated \
  --with 'litert-torch==0.9.4' --with tflite --with 'pydantic<2.12' \
  --with pyyaml --with loguru --with dulwich --with numpy \
  python scripts/export_cesta_firmware.py \
  --model runs/cesta/<run_id> --output firmware/model \
  --target node --receiver-index <node_index>
```

Export targets:

- `node` (default) — one graph receiver with its sender list; pass the device's graph index as `--receiver-index` and list the same senders in `NEIGHBORS` in `sender_indices` order.
- `local` — the shared per-sensor temporal encoder and classifier without communication.
- `graph` — centralized fixed-topology inference for a runtime that supplies every node window.

Export validates shapes, registered TensorFlow Lite Micro operators, and numerical parity with PyTorch, and writes `model.json` metadata beside `model.tflite`. The firmware parses `model.json` at boot and rejects exports whose target, receiver, sender list, shapes, or training status do not match `config.rs`.

## Build and flash

```bash
cd firmware
cargo check
cargo build --release
espflash flash target/xtensa-esp32s3-espidf/release/cesta-firmware --monitor
```

## Operation

Each diagnosis cycle runs the receiver-local CESTA contract on-device:

1. **Request pass** — encode the local window, threshold the exported request probabilities at the model's `request_threshold`, and send one binary request per neighbor listing the needed timesteps.
2. **Exchange** — neighbors answer from their most recent cached window with the requested hidden-state rows only (a zero-count response means no cached window yet). Requests and responses travel as fragmented ESP-NOW action frames (250-byte packets, reassembled per peer); `EXCHANGE_WAIT_MS` bounds the wait.
3. **Aggregate pass** — rerun the model with the received payloads and publish the aggregated diagnosis.

MQTT carries telemetry only; the neighbor exchange runs entirely over ESP-NOW.

## Telemetry

Nodes publish JSON to `cesta/readings/<device_id>`:

```json
{"device_id": "esp32_01", "timestamp": 1718000000, "temperature": 25.3, "humidity": 60.1, "path": "normal", "fault_mode": "normal", "gpio": 5}
{"device_id": "esp32_01", "timestamp": 1718000003, "type": "inference", "window_id": 12, "communication_mode": "gumbel_request", "label": "NORMAL", "class": 0, "confidence": 0.98, "probabilities": [0.98, 0.01, 0.0, 0.01], "requested": [["esp32_02", 7]], "received": [["esp32_02", 7]], "request_elapsed_ms": 210, "aggregate_elapsed_ms": 195}
```

A lab deployment can collect telemetry with Mosquitto, Telegraf, InfluxDB, and Grafana.

## Source layout

```text
src/
├── main.rs       # Sampling, diagnosis cycle, telemetry loop
├── config.rs     # Per-node static configuration (edit this)
├── fault.rs      # Fault profiles and software DRIFT/STUCK injectors
├── dht.rs        # DHT11 driver
├── inference.rs  # TFLite Micro node classifier wrapper
├── exchange.rs   # Binary request/response neighbor protocol
├── espnow.rs     # ESP-NOW transport with fragmentation
├── mqtt.rs       # Telemetry publisher
└── wifi.rs       # Station connection
model/            # Exported model.tflite + model.json (replace before flashing)
```
