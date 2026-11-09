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
| `EXCHANGE_CUTOFF_MS` / `CYCLE_BUDGET_MS` / `EXCHANGE_POLL_MS` | Nominal-bucket-close-relative exchange cutoff (40 s), cycle budget (60 s), and receiver poll interval (50 ms) |
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

Export targets:

- `node` (default): one graph receiver with its sender list; pass the device's graph index as `--receiver-index` and list the same senders in `NEIGHBORS` in `sender_indices` order.
- `local` and `graph` are rejected by the current firmware exporter: their probability-output wrappers do not satisfy the v3 distributed raw-logit/CRF contract.

## Common stage runtime and HiFiNet application

`components/tflite_micro` is the model-neutral TFLite Micro bridge used by CESTA through `src/tflite.rs`. Its 35-operator resolver includes `MEAN`, `RSQRT`, and `SQUARED_DIFFERENCE` for exported HiFiNet stages. It verifies the FlatBuffer and schema and requires one float32 input and output, each shaped `[1, 60, width]`.

- `StageRunner::new(arena_size)` creates an unloaded runner.
- `load(model_bytes, input_width, output_width)` destroys the previous interpreter, arena, and model copy before allocating the next stage and checks tensor widths. Failed loads leave the runner unloaded.
- `predict(input, output)` uses flattened caller-owned buffers, checks lengths and finite values, and copies the result out before a stage change.
- `unload()` and `Drop` release resources. Models are copied into aligned PSRAM, so source bytes need not outlive `load`. Allocation failures return errors.

Reuse one runner across sequential stages: it owns at most one interpreter and arena plus one model copy, not five arenas. Tensor allocation retains idle-priority scheduling; operator profiling yields periodically during invocation. The C API exports `tflite_micro_create`, `tflite_micro_destroy`, `tflite_micro_last_error`, `tflite_micro_input_count`, `tflite_micro_output_count`, and `tflite_micro_predict`. A creation error may return a handle that still requires destruction. CESTA retains its existing packed inputs, outputs, request logic, and CRF decoding.

HiFiNet is selected at build time; the default remains CESTA. `src/hifinet.rs` runs temporal → graph 0 → graph 1 → graph 2 → classifier through one runner, retains the original temporal embedding for the classifier, and decodes each timestep by first-index argmax without CRF. Raw temperature windows are never standardized. Exported graph stages include the always-available self loop; absent remote payloads are zero-filled and masked.

```bash
# From repository root; optional copied installation into an empty directory
uv run python scripts/install_hifinet.py \
  --bundle runs/hifinet-relocation-export-seed12-20261109T2106Z/bundle \
  --output firmware/model/hifinet

# From firmware; receiver must match config.rs NODE_INDEX and ordered NEIGHBORS
FIRMWARE_MODEL=hifinet \
HIFINET_BUNDLE="$PWD/model/hifinet" HIFINET_RECEIVER=3 \
cargo build
```

The build reruns bundle validation and embeds only the selected receiver's five stages. Validation covers stage hashes/lengths, declared operators/shapes, all receivers and graph edge order, three trained payload widths, cadence, preprocessing, decoder, and recorded conversion parity. Runtime verifies FlatBuffers and actual float32 tensor shapes while loading all five stages at startup; configuration mismatch disables inference. Compatibility identity hashes the canonical model/wire contract, topology/order, trained widths, preprocessing/decoder, provenance hashes, and sorted stage hashes/shapes across all receivers. Incidental source paths and validation results are excluded, so relocating an identical bundle does not break compatibility. Historic `missing_micro_operators` entries are accepted only when the current resolver now supports them.

HiFiNet exchange uses protocol version 4, rejects CESTA version 3 and different bundle fingerprints, and identifies each response by window, random nonce, and round in the nonce's high byte. Each window retains all three prior-stage caches until its common nominal-close-relative exchange cutoff so faster nodes can answer slower peers. The three round receive cutoffs divide that exchange budget equally; requests retry missing timesteps every two seconds. Payloads contain only that round's trained hidden width. Inference may continue with masked missing senders after a round cutoff; the common cycle deadline still discards late results. No previous window or different round payload is reused.

Benchmark telemetry retains the common schema-1 timing, communication edges, decision/window timestamps and prediction policy. Aggregated counts use `(round, edge, timestep)` identities: dense counts are `180 * neighbors`, and dense float payload bytes are `60 * neighbors * 4 * sum(round_widths)`. Per-edge requested/received byte totals use each round’s actual width; no single payload width is advertised across unequal rounds. Stage loading and invocation each check the common cycle deadline before admission and check again after execution. Benchmark telemetry reports every attempted round's per-edge requested/received timestep counts, retry/queue counts, width, cache admission and elapsed time, plus actual stage times and failures. Registered ESP-NOW peers are not claimed compatible merely because their MAC is registered. Inference explicitly labels zero received payloads `degraded_no_compatible_peer_payloads` (which also covers unreachable peers), and partial receipts `degraded_partial_round_payloads`. Even complete direct-neighbor receipts do not prove remote upstream graph completeness, so `full_graph_verified` remains false. A mixed CESTA/HiFiNet fleet cannot perform the full distributed HiFiNet graph; all participating nodes require matching bundles and node-specific builds.

The validated seed12 host bundle contains 14 stages totaling 846,724 model bytes. Host numerical parity and successful firmware compilation do not establish on-device numerical parity, heap headroom, radio delivery, or deadline performance; these require connected-board and multi-node validation. The CESTA synthetic diagnostic is not substituted for HiFiNet inference and is excluded from HiFiNet builds.

## Build and flash

```bash
cd firmware
cargo check
touch sdkconfig.defaults
cargo build --release
espflash flash target/xtensa-esp32s3-espidf/release/cesta-firmware --monitor
```

After C++ component edits, touch `sdkconfig.defaults` before rebuilding so `esp-idf-sys` reruns CMake; a Rust-only rebuild can retain a stale bridge library.

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
