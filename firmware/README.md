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
  python scripts/export_firmware.py \
  --model runs/cesta/<run_id> --output runs/cesta-export/bundle \
  --target node --receiver-index <node_index>
```

All exports use `bundle.json` and a `stages` table recording each FlatBuffer's filename, hash, length, shapes and operators. CESTA has one `node.tflite` stage; HiFiNet has `temporal.tflite`, `graph-<round>-node-<receiver>.tflite`, and `classifier.tflite`; STGCN has `prepare-<round>.tflite`, `graph-<round>-node-<receiver>.tflite`, `finish-<round>.tflite`, and `classifier.tflite`. Different stage counts and widths are architectural, not alternate bundle conventions.

Use `uv run --no-project --with tflite==2.18.0 python scripts/install_firmware.py --bundle <bundle> --architecture <model> --output <empty-directory>`. For CESTA, installation maps `bundle.json` to firmware-required `model.json` and `node.tflite` to `model.tflite`. Stage models copy the bundle unchanged. Installation refuses nonempty destinations; back up an existing CESTA model directory outside the destination before replacing it. No existing ignored artifacts are migrated automatically. Legacy CESTA `model.json`/`model.tflite` pairs are accepted with structural validation but cannot prove a historical model hash absent from their metadata. Existing staged v1 bundles remain supported. Existing boards and radio fingerprints are unchanged for identical contracts and model bytes; different checkpoints or stage bytes still require matching fleet deployments. The removed per-model command names are not retained as wrappers.

The same entrypoint detects HiFiNet and STGCN from `--model` checkpoint metadata. For these staged models, supply `--output <empty-bundle-directory>` and omit CESTA-only flags. `--architecture` optionally asserts the detected architecture. `--validate-only` runs staged PyTorch parity without conversion and does not create installable bundles. The per-model exporter and installer scripts have been removed; use only `export_firmware.py` and `install_firmware.py`.

CESTA export targets:

- `node` (default): one graph receiver with its sender list; pass the device's graph index as `--receiver-index` and list the same senders in `NEIGHBORS` in `sender_indices` order.
- `local` and `graph` are rejected by the current firmware exporter: their probability-output wrappers do not satisfy the v3 distributed raw-logit/CRF contract.

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
