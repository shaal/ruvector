# esp32-mmwave-sensor

ESP32-S3 firmware that reads a Seeed MR60BHA2 60 GHz mmWave radar over UART and emits parsed presence/heart-rate frames. Implements ADR-063 + RuView ADR-SYS-0024. Standalone Cargo project (not part of the workspace) using the `esp` Rust toolchain (Xtensa).

## Important files
- `Cargo.toml` — `ruvector-mmwave-sensor` bin; declares standalone `[workspace]`.
- `rust-toolchain.toml` — pins the `esp` Xtensa toolchain.
- `build.rs` — esp-idf build glue via `embuild`.
- `sdkconfig.defaults` / `sdkconfig.defaults.esp32s3` — ESP-IDF kconfig defaults.
- `src/main.rs` — firmware entry; sets up UART, runs the radar parse loop.
- `src/selftest.rs` — boot-time self-test.

## Run / build
- Requires `espup` + `cargo-espflash`.
- Build: `cargo build --release`.
- Flash: `cargo espflash flash --release --monitor`.

## Tech stack
- `esp-idf-svc` 0.51, `esp-idf-hal` 0.45, `esp-idf-sys` 0.36, `log`, `anyhow`.
- Shared parser: `../../crates/ruvector-mmwave` (no_std, zero-alloc).

## Related
- Host-side bridge in `../ruvLLM/esp32-flash` uses the same toolchain and shared crate.
