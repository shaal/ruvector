# ruvector-fpga-transformer

Ultra low-latency transformer inference with FPGA acceleration, coherence gating, and deterministic execution. Designed for embedded / hardware-support use cases with reproducible math and zero-allocation hot paths.

## Properties

- Deterministic latency paths (fixed shape inference with bounded timing).
- Quantization-first design: explicit INT4/INT8 with reproducible math.
- Zero-allocation hot path.
- Coherence gating with mincut integration.
- Multiple backends: FPGA PCIe, FPGA Daemon, Native Sim, WASM Sim.
- Witness logging integrated with ReasoningBank.

## Layout

- `Cargo.toml` — features include backend selection (`daemon`, `native_sim`, `pcie`, `wasm`), verification (`strict_verify`, `witness`), softmax options (`topk_only`, `lut_softmax`, `pwl_softmax`), `trace`. Defaults: `daemon`, `native_sim`, `witness`. Crypto via `sha2`, `ed25519-dalek` for artifact verification.
- `src/lib.rs` — public `Engine` + module re-exports (`artifact`, `backend`, `gating`, `quant`, `witness`, `types`, `error`); usage example shown in doc-comment.
- `src/error.rs` — crate errors.
- `src/types.rs` — `InferenceRequest`, `GateHint`, `FixedShape`, etc.
- `src/artifact/` — signed model-artifact format (manifest, pack, verify).
- `src/backend/` — backend implementations (PCIe / daemon / native_sim / wasm_sim).
- `src/gating/` — coherence + policy gates.
- `src/quant/` — quantization formats, calibration, LUTs.
- `src/witness/` — witness hashing + log (auditable inference trail).
- `src/ffi/` — C ABI + wasm-bindgen FFI surfaces.

## Tests / benches / examples

- `benches/`: `correctness.rs`, `gating.rs`, `latency.rs`.
- `examples/`: `basic_inference.rs`, `daemon_client.rs`.
- No `tests/` folder; correctness lives in `benches/correctness.rs`.

## Related crates

- `crates/prime-radiant` — provides the coherence math behind the gate.
- `crates/ruvector-mincut` — feeds gate decisions.
