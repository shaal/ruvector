# micro-hnsw-wasm/src

Single source file implementing the neuromorphic HNSW kernel.

- `lib.rs` — `#![no_std]`. Defines vector / metric / node types, the HNSW graph and search, beam search + multi-core result merging, the SNN layer (LIF neurons, STDP, winner-take-all, dendritic non-linearity, homeostatic plasticity, gamma oscillator), and the `#[no_mangle] extern "C"` WASM exports.

Compile-time knobs at the top of the file: `MAX_VECTORS`, `MAX_DIMS`, `MAX_NEIGHBORS`, `BEAM_WIDTH`, plus all SNN time constants.

See parent `../CLAUDE.md` for the crate-level overview.
