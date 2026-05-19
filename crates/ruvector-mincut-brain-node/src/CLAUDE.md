# ruvector-mincut-brain-node/src

Single-file source — small by design (it produces a tiny `cdylib`).

## Files

- `lib.rs` — `pub use ruvector_mincut::wasm::canonical::*;` plus V1 ABI stubs (`malloc`, `feature_extract_dim`, `feature_extract`). Bump allocator state lives in a static `AtomicU32` starting at 64KB.

## Notes

- `feature_extract*` stubs are intentional no-ops — this is a graph-algorithm node, not an embedding node.
- Linear `memory` is auto-exported by the WASM linker; no Rust source is needed for it.
