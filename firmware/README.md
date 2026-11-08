# Firmware

ESP32-S3 Rust firmware for distributed CESTA diagnosis. Each node samples a DHT11 sensor, encodes a local 60-sample window, exchanges selected neighbor hidden states over ESP-NOW, and publishes readings and per-timestep `NORMAL`/`SPIKE`/`DRIFT`/`STUCK` diagnoses over MQTT.

## Hardware and toolchain requirements

- ESP32-S3 board with **octal PSRAM** and at least **8 MB flash**.
- Pin ESP-IDF v5.3.6. Configure 8 MB QIO flash and octal PSRAM at 40 MHz, routing large allocations to PSRAM through `sdkconfig.defaults`.
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

### Recorded boards

| Assigned device | Node index | Factory base MAC | Detected flash | Verified |
|---|---|---|---|---|
| `esp32_01` | 0 | `E0:72:A1:D5:5E:78` | 16 MB | `espflash board-info`, 2026-11-07, ESP32-S3 revision v0.2 |
| `esp32_02` | 1 | `28:84:85:65:66:A0` | 16 MB | `espflash board-info`, 2026-11-07, ESP32-S3 revision v0.2 |
| `esp32_03` | 2 | `E0:72:A1:D8:F9:9C` | 16 MB | `espflash board-info`, 2026-11-07, ESP32-S3 revision v0.2 |
| `esp32_04` | 3 | `E0:72:A1:D6:CA:A8` | 16 MB | `espflash board-info`, 2026-11-07, ESP32-S3 revision v0.2 |


## Fault profiles

`src/config.rs` retains `FAULT_NORMAL` as the default. Set `FAULT_CONFIG = FAULT_ALL` to enable mixed fault injection.

- SPIKE: a constant, randomly signed offset per event.
- DRIFT: a linear offset with a randomly selected rate and direction per event.
- STUCK: exactly the first temperature of the event, with **zero jitter**.

## Model export

Export a trained checkpoint before flashing. The exporter checks numerical parity and registered operators and writes metadata beside the model.

```bash
# From the repository root
uv run --isolated \
  --with 'litert-torch==0.9.4' --with tflite --with 'pydantic<2.12' \
  --with pyyaml --with loguru --with dulwich --with numpy \
  python scripts/export_cesta_firmware.py \
  --model runs/cesta/<run_id> --output firmware/model \
  --target node --receiver-index <node_index>
```

The validated DHT seed-12 CRF0.02 exports are in `runs/firmware-export-cesta-v3-seed12-20261107T154717Z/node-0/` through `node-3/`. Receiver 0 is installed in `firmware/model/`; the previous Intel artifact is preserved under that run's `firmware-model-backup/`. Each node requires its own matched binary/metadata pair and sender order. Desktop parity does not establish on-device latency, arena headroom, or radio delivery.

Export targets:

- `node` (default): one graph receiver with its sender list; pass the device's graph index as `--receiver-index` and list the same senders in `NEIGHBORS` in `sender_indices` order.
- `local` and `graph` are rejected by the current firmware exporter: their probability-output wrappers do not satisfy the v3 distributed raw-logit/CRF contract.

### Inference and exchange contracts

- Embed `model/model.tflite` and invoke Espressif TensorFlow Lite Micro through the `components/cesta_tflite` C++ bridge.
- Require v3 metadata and raw class logits, with checkpoint CRF transitions retained for firmware Viterbi decoding. Reject legacy probability-only exports and mismatched receiver, sender order, shapes, architecture, or training status. Softmax probabilities are emission probabilities, not CRF marginals.
- Preserve receiver-local requests, received-neighbor attention, fusion/correction, and full-window decoding. MQTT reports all decoded timestep labels as well as the final timestep's diagnosis; decoded classes need not equal independent softmax argmax.
- Use binary request/response exchanges directly over ESP-NOW. `espnow.rs` fragments frames into 250-byte packets and reassembles them per peer; its worker serves cached hidden-state rows for requested timesteps and sends queued frames.
- Peers use validated static `NEIGHBORS` station MAC addresses, unencrypted, on the current station channel. Unknown identities and invalid or duplicate MACs are rejected. Incoming model-neighbor configuration must be reciprocal for response routing; the DHT graph satisfies this. MAC/name binding is not cryptographic authentication. MQTT carries telemetry only.
- Replace all zero MAC placeholders with actual station addresses before deployment. Receiver 0 uses DHT sender indices 2 and 3 (`esp32_03` and `esp32_04`). Registered-peer masks represent valid local radio registrations, not proof of RF reachability; absent responses remain masked out of attention.
- Inference uses shared SNTP-aligned one-minute slots and complete 60-sample windows. Failed reads, missed slots, or reads over two seconds late invalidate the window rather than compressing time. Sampling waits for synchronization; blocking inference overruns trigger fresh warmup. This conservative policy skips incomplete windows instead of reproducing offline masked-window inference.
- Exchange identities include actual window timestamps, shared checkpoint SHA256, and request nonces. Never substitute a stale cached window for a requested one or combine hidden states from different checkpoints. Early requests wait for matching cache readiness within a bounded deadline; replies are admitted only for the active request cycle. Queues and reassembly are bounded; send completion and pacing do not guarantee successful radio delivery. Failed frames are abandoned, with new requests possible in subsequent inference cycles.
- Threshold requests at the model's `request_threshold` (evaluation semantics), wait up to `EXCHANGE_WAIT_MS`, then rerun inference with received payloads.
- Preserve watchdog servicing during long float32 passes: the bridge yields at operator boundaries and temporarily lowers the caller to idle priority during tensor allocation, then restores it.
- In `main.rs`, construct SNTP from `NTP_SERVER`, not `EspSntp::new_default()`.

## Build and flash

```bash
cd firmware
cargo check
touch sdkconfig.defaults
cargo build --release
espflash flash target/xtensa-esp32s3-espidf/release/cesta-firmware --monitor
```

`partitions.csv` reserves a 6 MiB factory application partition at `0x10000` for the embedded model on 8 MB flash; this layout does not provide OTA slots. Existing generated sdkconfig files can retain old 2 MB flash or small application partitions despite changes to defaults. Build with a fresh generated sdkconfig, then verify the resolved flash size, custom partition selection, PSRAM settings, and offline image fit before flashing. Preserve existing device data when changing partition layouts.

After C++ component edits, touch `sdkconfig.defaults` before rebuilding so `esp-idf-sys` reruns CMake; a Rust-only rebuild can retain a stale bridge library.

### Memory safeguards

- Select `CONFIG_ESPTOOLPY_FLASHMODE_QIO=y` and `CONFIG_SPIRAM_MODE_OCT=y` independently. The derived `CONFIG_ESPTOOLPY_FLASHMODE="dio"` and ROM image header are intentional; do not manually change them to `"qio"`.
- Keep `CONFIG_SPIRAM_MEMTEST=y`, including during startup diagnosis or memory-timing changes. PSRAM identification alone does not validate memory access.
- Match the flashing tool's flash frequency to the compiled configuration. After memory-configuration changes, verify the resolved build configuration and both normal and synthetic firmware modes.

### Troubleshooting: PSRAM startup crash

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

Use `target/xtensa-esp32s3-espidf/release/build/esp-idf-sys-<build-id>/out/build/bootloader/bootloader.bin`. This project-built bootloader's QIO initialization resolved the tested N16R8 startup failure.

### Synthetic diagnostics and console

Normal builds run the sensor/radio application. Enable synthetic diagnosis only through `INFERENCE_SYNTHETIC_DIAGNOSTIC` in `src/config.rs`, not a Cargo feature. Keep it `false` by default and restore it after testing.

Synthetic mode logs a 15-second startup countdown, runs its cases once, and repeats the final PASS/FAIL every five seconds for serial reconnection. It does not start the normal sensor or radio loop.

Use UART0 at 115200 baud as the primary console and native USB Serial/JTAG as secondary. Prefer the USB-UART bridge for startup and panic capture: ESP32 resets can disconnect native USB and its USB/IP forwarding.

## Spatial-model measurement protocol

Compare CESTA, ST-GCN, DCRNN, and HiFiNet on identical input/label streams, sampling, window policy, and neighbor availability. The paper places HMCT among node-local temporal baselines. This firmware executes CESTA only; a common MQTT record format does not provide baseline inference implementations. Device and server-side execution results must remain separate.

| Metric | Measurement and interpretation |
|---|---|
| Memory | Internal/PSRAM free and minimum-free heap, largest free blocks, configured tensor arena; snapshots are not exact per-model peak allocation |
| Latency | Request inference, exchange processing, aggregate inference, and full cycle; mean, median, p95, p99, and maximum |
| Decision delay | Latest nominal sample-slot to decision and window-ready to decision; wall-clock measurements require synchronization |
| Modeled radio energy | Explicit distances and common first-order radio constants applied to logical payload counts; not measured ESP32 or battery energy |

`cycle_us` includes peer setup, both prediction passes (including decoding), and exchange processing. It excludes acquisition, window insertion, telemetry serialization, and MQTT publication. `exchange_wait_us` includes request construction, queueing, polling, and response processing, not just radio-active time. Failed attempts retain completed/partial stage timings separately; unfinished stages are null.

Online reports are restricted to these four metric families; accuracy remains an offline benchmark. Logical edge counts and payload width are retained only as energy-calculation inputs, not as a separate reported communication metric. Deduplicate MQTT QoS retransmissions by boot/device/event identity. Keep run and stream identities explicit; independently randomized live fault injection is not an identical replay across models. Retain failure/audit information to avoid presenting only successful-cycle performance. Actual energy requires an external power measurement synchronized to the run; unavailable measurements must not be reported as zero.

## Telemetry

Nodes publish JSON to `cesta/readings/<device_id>`. Inference records include per-neighbor timestep counts (`requested`/`received`) and pass latencies:

```json
{"device_id": "esp32_01", "timestamp": 1718000000, "temperature": 25.3, "humidity": 60.1, "path": "normal", "fault_mode": "normal", "gpio": 5}
{"device_id": "esp32_01", "timestamp": 1718000003, "type": "inference", "window_id": 12, "communication_mode": "gumbel_request", "label": "NORMAL", "class": 0, "confidence": 0.98, "probabilities": [0.98, 0.01, 0.0, 0.01], "requested": [["esp32_02", 7]], "received": [["esp32_02", 7]], "request_elapsed_ms": 210, "aggregate_elapsed_ms": 195}
```

Lab deployments can collect telemetry with Mosquitto, Telegraf, InfluxDB, and Grafana.
