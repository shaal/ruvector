# ruvector-mincut-gated-transformer-wasm

WASM bindings for the mincut-gated transformer (`ruvector-mincut-gated-transformer`). Provides ultra-low-latency, deterministic-bound inference with explainable per-step witnesses, gated by the dynamic minimum-cut coherence signal.

## Features

- `default = ["console_error_panic_hook"]`

## Layout

- `Cargo.toml` — `cdylib` + `rlib`; path dep on `ruvector-mincut-gated-transformer` (with `wasm` feature, no default features); `wasm-bindgen`, `serde-wasm-bindgen`, `js-sys`.
- `src/lib.rs` — single source file with `WasmTransformer` and `WasmGatePacket` bindings.
- `examples/web_scorer.rs` — runnable web scorer using the WASM transformer.
- `tests/web.rs` — `wasm-bindgen-test` integration test.

## Public API (WASM)

- `init()` — `#[wasm_bindgen(start)]`.
- `WasmTransformer::new()` then `infer(tokens: Uint32Array, gate: WasmGatePacket) -> WasmInferResult`.
- `WasmGatePacket` — `{ lambda, lambda_prev, boundary_edges, boundary_concentration_q15, partition_count }`.
- Wrapped types from the core crate: `GateDecision`, `GatePacket`, `GatePolicy`, `GateReason`, `InferInput`, `InferOutput`, `MincutGatedTransformer`, `QuantizedWeights`, `SpikePacket`, `TransformerConfig`.

## Related

- `crates/ruvector-mincut-gated-transformer` — underlying transformer.
- `crates/cognitum-gate-kernel` — produces the witness fragments aggregated into the gate packet.
- `crates/ruvector-dag/src/attention/mincut_gated.rs` — DAG-level mincut-gated attention.
