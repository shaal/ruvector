# ruvector-fpga-transformer/examples

## Files

- `basic_inference.rs` — minimal `Engine + NativeSimBackend + DefaultCoherenceGate` setup and a single inference call.
- `daemon_client.rs` — client wiring against the FPGA daemon backend (requires `daemon` feature).

Run via `cargo run --example <name> -p ruvector-fpga-transformer`.
