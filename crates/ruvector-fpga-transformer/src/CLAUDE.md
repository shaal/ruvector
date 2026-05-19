# ruvector-fpga-transformer/src

FPGA-transformer crate source root.

## Top-level files

- `lib.rs` — crate doc + module wiring; defines the public `Engine`.
- `error.rs` — crate errors.
- `types.rs` — `InferenceRequest`, `GateHint`, `FixedShape`, and other value objects.

## Subdirectories

- `artifact/` — signed model-artifact format (manifest + pack + verify).
- `backend/` — per-target backends (FPGA PCIe, FPGA daemon, native_sim, wasm_sim).
- `gating/` — coherence + policy gating.
- `quant/` — quantization formats, calibration, LUTs.
- `witness/` — auditable witness hashing + log.
- `ffi/` — C ABI + wasm-bindgen surfaces.
