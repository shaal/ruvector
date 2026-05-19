# micro-hnsw-wasm

Single-file `no_std` neuromorphic HNSW vector search compiled to an ~12 KB WASM module for edge AI, ASIC, and embedded systems. Combines a classical HNSW graph (multi-core sharded — 256 cores × 32 vectors = 8 K total) with LIF spiking neurons, STDP learning, winner-take-all, dendritic computation, homeostatic plasticity, and gamma-oscillator resonance.

## Layout

- `Cargo.toml` — standalone (`[workspace]` reset). Builds as `cdylib` only, no runtime dependencies; aggressive size optimisation (`opt-level = "z"`, LTO, `panic = "abort"`, strip).
- `src/lib.rs` — entire implementation in one file (~hundreds of lines): metric enum (`L2`, `Cosine`, `Dot`), `Vector { data: [f32; MAX_DIMS] }`, multi-core HNSW with beam search and result merging, SNN integration (LIF/STDP/WTA), Cypher-style typed node graph (16 node types), edge weights for GNN message passing, vector updates for online learning.
- `verilog/micro_hnsw.v` — Verilog mirror of the algorithm for ASIC / FPGA targets; same kernel, different toolchain.
- `micro_hnsw.wasm` — checked-in built artifact for quick consumption.
- `benchmark.js`, `test_wasm.js` — Node smoke tests / micro-benchmarks for the WASM binding.
- `DEEP_REVIEW.md` — design + neuromorphic-feature commentary.
- `Cargo.lock` — committed because this crate is excluded from the workspace.

## Configuration constants (src/lib.rs)

`MAX_VECTORS = 32`, `MAX_DIMS = 16`, `MAX_NEIGHBORS = 6`, `BEAM_WIDTH = 3`; SNN constants `TAU_MEMBRANE`, `TAU_REFRAC`, `STDP_A_PLUS/MINUS`, `WTA_INHIBITION`, etc.

## Public API

C-style WASM exports for insert/search/update plus the SNN-extended variants. Consumed by `benchmark.js` / `test_wasm.js`.

## Related

- `crates/ruvector-cnn-wasm`, `crates/ruvector-verified-wasm`, `crates/ruvector-mincut-gated-transformer-wasm` — companion size-optimised WASM modules in the same family.
- `npm/` packages distribute many of these as `@ruvector/...` modules.
