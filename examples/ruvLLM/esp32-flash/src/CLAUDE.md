# ruvLLM / esp32-flash / src

Firmware source for `ruvllm-esp32-flash`. Tracks `../../esp32/src/` closely, with extra plumbing for OTA, diagnostics, and the flashable binary entry.

## Top-level files
- `lib.rs` - crate root (library = `ruvllm_esp32`).
- `main.rs` - firmware entry (the `ruvllm-esp32` binary registered in `../Cargo.toml`).
- `benchmark.rs` - on-device benchmark routines.
- `diagnostics.rs` - hardware diagnostics output.
- `ota.rs` - over-the-air update logic.

## Subdirectories
- `federation/` - multi-chip pipeline / protocol / speculative-decoding code.
- `models/` - model catalog and loaders.
- `optimizations/` - quantization, lookup tables, pruning, sparse attention, micro-LoRA.
- `ruvector/` - micro-HNSW, RAG, semantic memory, anomaly detection.

## Build
- ESP32: `make build` (from `../`) with ESP-IDF env loaded.
- Host: `cargo build --features host-test`.

## Related
- Library-only sibling: `../../esp32/src/`. NPM wrapper: `../npm/`.
