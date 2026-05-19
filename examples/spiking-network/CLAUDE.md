# spiking-network

Event-driven spiking neural network library optimised for ASIC deployment. The philosophy is "compute only on change": neurons stay silent until a threshold is crossed, so the model is sparse, event-driven, and friendly to a low-power ASIC routing fabric.

## Important files
- `Cargo.toml` - library crate (plus three `[[example]]`s). Deps: `ruvector-core` + `ruvector-gnn`, `ndarray`, `rand` / `rand_distr`, `rayon`, `parking_lot`, `dashmap`, `indexmap`, `smallvec`, `bitvec`, `priority-queue`. Features `default = ["simd"]`, plus `wasm` and `visualization`. README at `docs/README.md` (not in this chunk).
- `src/lib.rs` - crate root + library-level documentation on the philosophy and architectural benefits.
- `src/error.rs` - error type.
- `src/neuron/`, `src/network/`, `src/encoding/` - the three core subsystems (see CLAUDE.md inside each).

## Examples / build
- Examples (cargo `[[example]]`): `edge_detection`, `pattern_recognition`, `asic_simulation`. They live at `src/examples/` (path declared in `Cargo.toml`, not within this chunk).
- `cargo build -p spiking-network` (release: add `--release`).
- `cargo bench -p spiking-network` (Criterion bench `spiking_bench`).

## Related
- Workspace `ruvector-core`, `ruvector-gnn` (used as path deps).
- Brain-related research demos: `../brain-boundary-discovery/`, `../real-eeg-analysis/`.
