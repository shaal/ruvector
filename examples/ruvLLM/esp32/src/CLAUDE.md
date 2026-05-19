# ruvLLM / esp32 / src

Source for the `ruvllm-esp32` crate (`no_std`-friendly).

## Top-level files
- `lib.rs` - crate root with feature-gated sub-modules.
- `main.rs` - default firmware entry (run on ESP32 with ESP-IDF; reverts to a host main when `host-test` is enabled).
- `attention.rs`, `embedding.rs`, `micro_inference.rs`, `model.rs`, `quantized.rs` - inference core.
- `benchmark.rs`, `diagnostics.rs` - on-device benches + diagnostics.
- `ota.rs` - OTA update support.

## Subdirectories
- `federation/` - multi-chip coordination, sharding, speculative decoding, tensor-parallel, FastGRNN router.
- `models/` - shipped model metadata / loaders.
- `optimizations/` - binary/product quantization, lookup tables, micro-LoRA, sparse attention, pruning.
- `ruvector/` - on-device vector layer (micro-HNSW, RAG, semantic memory, anomaly, federated search, hyperbolic).

## Build
- `cargo build` from `../` (host). ESP32 target: ESP-IDF toolchain via `idf.py` / cargo-espflash.

## Related
- Companion flashable crate: `../../esp32-flash/src/`. Higher-level ruvLLM: `../../src/`.
