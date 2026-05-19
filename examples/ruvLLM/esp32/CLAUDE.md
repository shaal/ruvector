# ruvLLM / esp32

`ruvllm-esp32` - tiny LLM inference for ESP32 microcontrollers. `no_std`-friendly crate with INT8/INT4 quantization, multi-chip federation, RuVector semantic memory (micro-HNSW + RAG), and SNN-gated energy optimization. Standalone Cargo workspace (`[workspace]` block) - not part of the main repo workspace.

## Important files
- `Cargo.toml` - standalone crate. ESP-IDF deps (`esp-idf-svc/-hal/-sys`) are optional; `heapless`, `libm`, `fixed`, `postcard`, `serde` are the embedded core. `anyhow` is gated for host testing.
- `Cargo.lock` - committed lockfile.
- `src/` - library + main binary plus subsystems for inference, quantization, federation, RuVector micro-store.
- `examples/` - 14 standalone cargo examples (anomaly_industrial, classification, embedding_demo, federation_demo, massive_scale_demo, medium_scale_demo, model_sizing_demo, optimization_demo, rag_smart_home, snn_gated_inference, space_probe_rag, swarm_memory, user_demo, voice_disambiguation).
- `benches/esp32_simulation.rs` - host-side benchmark that simulates ESP32 constraints.
- `tests/simulation_tests.rs` - host-side simulation tests.

## Build / run
- Host tests: `cd examples/ruvLLM/esp32 && cargo test --features anyhow`.
- ESP32 build: requires the ESP-IDF toolchain (`esp-idf-sys 0.35`, `esp-idf-hal 0.44`, `esp-idf-svc 0.49`); see top-level README in the crate.

## Related
- Flashable companion with more wiring (Makefile, sdkconfig, Docker, web flasher): `../esp32-flash/`.
- Host-side ruvLLM crate: `../`. RuVector core: `../../crates/ruvector-core`.
