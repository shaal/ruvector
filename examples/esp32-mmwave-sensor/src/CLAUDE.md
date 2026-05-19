# esp32-mmwave-sensor/src

Firmware source for the ESP32-S3 mmWave radar reader.

## Important files
- `main.rs` — boot, UART setup, main parse loop using `ruvector-mmwave`.
- `selftest.rs` — startup self-test routines logged via `log`.

## Build
- From parent: `cargo build --release` (requires `esp` toolchain).
