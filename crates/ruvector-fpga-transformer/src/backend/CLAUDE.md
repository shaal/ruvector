# ruvector-fpga-transformer/src/backend

Backend implementations of the inference trait. Selected at runtime; each is feature-gated.

## Files

- `mod.rs` — `Backend` trait + dispatch.
- `fpga_pcie.rs` — direct PCIe FPGA access (feature `pcie`).
- `fpga_daemon.rs` — talks to a local daemon over UDS / tokio (feature `daemon`).
- `native_sim.rs` — pure-Rust deterministic simulator (feature `native_sim`; the reference path).
- `wasm_sim.rs` — WASM-compatible simulator (feature `wasm`).
