# ruvector-sparse-inference/src/model

Model loading and per-architecture runners.

- `mod.rs` — module roots, common model traits.
- `gguf.rs` — GGUF reader (byteorder/half), see `docs/GGUF_IMPLEMENTATION.md`.
- `loader.rs` — generic memory-mapped loader (memmap2).
- `runners.rs` — per-architecture inference runners (Llama-like FFN/attention).
- `types.rs` — shared model types (tensor headers, layer descriptors).
