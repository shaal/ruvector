# ruvLLM / esp32-flash

`ruvllm-esp32-flash` - the flashable, full-featured ESP32 deployment of ruvLLM. Includes RAG, federation, WASM bindings, OTA, scripts for Mac/Linux/Windows, a Dockerised toolchain, and an `npm` distribution that ships precompiled firmware + a CLI flasher + a web flasher. Standalone Cargo workspace, `publish = false`.

## Important files
- `Cargo.toml` - standalone crate. Pinned ESP-IDF stack (`esp-idf-svc 0.51.0`, `esp-idf-hal 0.45.2`, `esp-idf-sys 0.36.1`); features `esp32`, `wasm`, `host-test`, `federation`, `full`.
- `Cargo.lock` - committed.
- `build.rs` - ESP-IDF build glue.
- `Makefile`, `Dockerfile`, `sdkconfig.defaults`, `sdkconfig.defaults.esp32s3` - toolchain + ESP32 / ESP32-S3 configs.
- `cluster-flash.{sh,ps1}`, `cluster-monitor.sh`, `cluster.example.toml`, `flash-windows.bat`, `install.{sh,ps1}` - multi-device flashing / monitoring scripts.
- `src/` - library + main binary, federation, models, optimizations, ruvector subdirs (see CLAUDE.md inside).
- `scripts/`, `scripts/windows/` - shell + PowerShell helpers (also mirrored under `npm/`).
- `web-flasher/index.html` - in-browser ESP Web Tools flasher.
- `npm/` - `ruvllm-esp32` npm package distribution (CLI + scripts + web flasher).

## Build / flash
- Native firmware build (with ESP-IDF env): `make build` then `make flash`.
- Docker: `docker build -t ruvllm-esp32 . && docker run ...`.
- Via npm: `npx ruvllm-esp32` (see `npm/`).
- Host tests: `cargo test --features host-test`.

## Related
- Library-only sibling: `../esp32/`. Host-side: `../`.
