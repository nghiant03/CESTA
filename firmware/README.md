# Firmware

ESP32-S3 firmware that turns each board into a distributed CESTA node. Every node samples a DHT11 sensor, encodes its local 60-sample window with an exported TensorFlow Lite Micro model, requests selected hidden-state payloads from its neighbors directly over ESP-NOW, aggregates the replies, and publishes readings and per-timestep `NORMAL`/`SPIKE`/`DRIFT`/`STUCK` diagnoses over MQTT.

## Hardware and toolchain requirements

- ESP32-S3 board with **octal PSRAM** and at least **8 MB flash**. 
- Host tools: [`espup`](https://github.com/esp-rs/espup), [`espflash`](https://github.com/esp-rs/espflash), `ldproxy`, and the ESP-IDF host dependencies.

## Configuration

One firmware image is built per deployed node. Edit `src/config.rs`:

| Setting | Purpose |
|---|---|
| `WIFI_SSID` / `WIFI_PASSWORD` | Station credentials (ESP-NOW rides the station interface) |
| `MQTT_SERVER` / `MQTT_PORT` / `MQTT_USER` / `MQTT_PASSWORD` | Telemetry broker |
| `DEVICE_ID` | Unique node name |
| `NODE_INDEX` | Graph node index |
| `NEIGHBORS` | Graph senders in `sender_indices` order: `device_id`, `node_index`, and station MAC |
| `DHT_PIN` | Sensor pin |
| `INFERENCE_ENABLED` / `INFERENCE_TENSOR_ARENA_BYTES` | Toggle and size the TFLite Micro arena |
| `INFERENCE_SYNTHETIC_DIAGNOSTIC` | Set `true` for serial-only synthetic testing; keep `false` for normal operation |
| `NTP_SERVER` and timing | SNTP clock sync for timestamps |
| `EXCHANGE_WAIT_MS` / `EXCHANGE_POLL_MS` | Diagnosis-cycle exchange deadline and worker poll interval |
| `FAULT_CONFIG` | Fault profile  |

## Fault profiles

`FAULT_CONFIG` supports one profile per fault type in the main codebase:

- `FAULT_NORMAL` — unmodified readings.
- `FAULT_SPIKE` — constant offset per event with magnitude sampled from `spike_magnitude_range` °C and a random sign, over consecutive events of `event_duration_samples` samples.
- `FAULT_DRIFT` — linear drift of `drift_rate` °C per sample with a random direction per event, over consecutive events of `event_duration_samples` samples.
- `FAULT_STUCK` — freezes at the first reading of each event with Gaussian jitter of `jitter_std` °C.

## Model export

Export a trained checkpoint before flashing.

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

## Build and flash

```bash
cd firmware
cargo check
touch sdkconfig.defaults
cargo build --release
espflash flash target/xtensa-esp32s3-espidf/release/cesta-firmware --monitor
```

### Troubleshooting: Crash during PSRAM startup

If PSRAM startup crashes, edit `firmware/sdkconfig.defaults`. Add/uncomment:

```ini
CONFIG_ESPTOOLPY_FLASHMODE_QIO=y
CONFIG_ESPTOOLPY_FLASHFREQ_40M=y
CONFIG_SPIRAM_SPEED_40M=y
CONFIG_ESPTOOLPY_FLASHSIZE_8MB=y
```

Comment out competing selections if present:

```ini
# CONFIG_ESPTOOLPY_FLASHMODE_DIO=y
# CONFIG_ESPTOOLPY_FLASHMODE_QOUT=y
# CONFIG_ESPTOOLPY_FLASHMODE_DOUT=y
# CONFIG_ESPTOOLPY_FLASHFREQ_80M=y
# CONFIG_SPIRAM_SPEED_80M=y
```

Keep `CONFIG_SPIRAM=y`, `CONFIG_SPIRAM_MODE_OCT=y`, `CONFIG_SPIRAM_BOOT_INIT=y`, and `CONFIG_SPIRAM_MEMTEST=y`. QIO selects **flash** mode, not PSRAM mode. Rebuild from `firmware/` and flash the matching bootloader:

```bash
touch sdkconfig.defaults
cargo build --release
espflash flash --port <serial-port> --bootloader <bootloader.bin> \
  --flash-size 8mb --flash-mode dio --flash-freq 40mhz \
  --monitor target/xtensa-esp32s3-espidf/release/cesta-firmware
```

Use `target/xtensa-esp32s3-espidf/release/build/esp-idf-sys-<build-id>/out/build/bootloader/bootloader.bin`.

## Telemetry

Nodes publish JSON to `cesta/readings/<device_id>`:

```json
{"device_id": "esp32_01", "timestamp": 1718000000, "temperature": 25.3, "humidity": 60.1, "path": "normal", "fault_mode": "normal", "gpio": 5}
{"device_id": "esp32_01", "timestamp": 1718000003, "type": "inference", "window_id": 12, "communication_mode": "gumbel_request", "label": "NORMAL", "class": 0, "confidence": 0.98, "probabilities": [0.98, 0.01, 0.0, 0.01], "requested": [["esp32_02", 7]], "received": [["esp32_02", 7]], "request_elapsed_ms": 210, "aggregate_elapsed_ms": 195}
```
