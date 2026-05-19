# spiking-network / src

Rust source for the spiking-network library.

## Important files
- `lib.rs` - crate root + library docs covering the event-driven philosophy and ASIC-friendly design.
- `error.rs` - shared error type (`thiserror`).

## Subdirectories
- `neuron/` - neuron models (LIF, Izhikevich) behind a common trait.
- `network/` - the spiking network itself: topology, event scheduler, simulation loop.
- `encoding/` - input spike encoders (rate / temporal / population).

## Build
- `cargo build -p spiking-network` (default features include `simd`).

## Related
- Examples referenced by `../Cargo.toml`: `src/examples/edge_detection.rs`, `pattern_recognition.rs`, `asic_simulation.rs` (the `src/examples/` directory itself is outside this chunk).
