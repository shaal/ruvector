# ruvLLM / esp32 / examples

Standalone cargo examples for `ruvllm-esp32`. Each file is buildable on its own and exercises a different ESP32 subsystem.

## Important files
- `anomaly_industrial.rs` - industrial anomaly detection over the micro vector store.
- `classification.rs` - tiny classifier demo.
- `embedding_demo.rs` - on-device embedding generation.
- `federation_demo.rs` - multi-chip federation (`src/federation/`).
- `massive_scale_demo.rs`, `medium_scale_demo.rs`, `model_sizing_demo.rs` - scale/size sweeps.
- `optimization_demo.rs` - quantization / lookup tables / pruning showcase.
- `rag_smart_home.rs`, `space_probe_rag.rs` - RAG flows for very different deployment scenarios.
- `snn_gated_inference.rs` - SNN-gated energy optimization.
- `swarm_memory.rs` - shared swarm memory.
- `user_demo.rs` - end-user-facing walkthrough.
- `voice_disambiguation.rs` - small voice classifier disambiguation.

## Run
- `cargo run --example <name>` from `../` (host) - most are designed for the simulation harness; ESP32 deployment requires the ESP-IDF toolchain.

## Related
- Library being exercised: `../src/`. Flashable companion: `../../esp32-flash/`.
