# ruvector-hailo/models

Drop-in directory for runtime artifacts consumed by `HailoEmbedder::open`:

- `model.hef` — compiled HEF (Hailo Executable Format) for the NPU path (feature `hailo`).
- `model.safetensors` + `tokenizer.json` — CPU-fallback artifacts (feature `cpu-fallback`).

Empty / gitignored by default. SHA256 of the HEF can be pinned via the `RUVECTOR_HEF_SHA256` env var; verification lives in `src/hef_verify.rs`.
