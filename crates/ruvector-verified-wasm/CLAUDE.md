# ruvector-verified-wasm

WASM bindings for `ruvector-verified` — proof-carrying vector operations in the browser. Provides a `JsProofEnv` that proves dimension equality, verifies batches of vectors, exports statistics, and produces compact (~82-byte) attestations.

## Layout

- `Cargo.toml` — `cdylib` + `rlib`; path dep on `ruvector-verified` (with `ultra` feature); `wasm-bindgen`, `serde-wasm-bindgen`, `js-sys`, `web-sys` (`console`).
- `src/lib.rs` — main bindings: `init`, `version`, `JsProofEnv`.
- `src/utils.rs` — `set_panic_hook`, `console_log` helpers.
- `tests/web.rs` — `wasm-bindgen-test` integration test for the browser API.

## Public API (WASM)

- `init()` — automatic `[wasm_bindgen(start)]`.
- `version() -> String`.
- `JsProofEnv::new()`, `prove_dim_eq(a, b) -> proofId`, `verify_batch(dim, Float32Array[])`, `stats()`, `create_attestation(proofId) -> { bytes: Uint8Array }` (~82 bytes).
- Underlying types: `ProofEnvironment`, `cache::ConversionCache`, `fast_arena::FastTermArena`, `gated::{ProofKind, ProofTier}`, `proof_store`, `vector_types`.

## Related

- `crates/ruvector-verified` — Rust library with the proof system; this crate is its WASM-facing facade.
- `crates/micro-hnsw-wasm`, `crates/ruvector-cnn-wasm`, `crates/ruvector-learning-wasm`, `crates/ruvector-mincut-gated-transformer-wasm` — sibling WASM modules.
