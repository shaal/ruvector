# ruqu-algorithms

Production-ready quantum algorithms in Rust built on the `ruqu-core` simulation engine. Provides VQE (chemistry), Grover's search, QAOA (optimization), and Surface-Code error correction.

## Layout

- `Cargo.toml` — workspace metadata, depends on `ruqu-core`, `rand`, `thiserror`, optional `serde` + `tracing`. Research-tier lint allow-list.
- `src/lib.rs` — re-exports `{run_vqe, VqeConfig, VqeResult}`, `{run_grover, GroverConfig, GroverResult}`, `{run_qaoa, QaoaConfig, QaoaResult, Graph}`, `{run_surface_code, SurfaceCodeConfig, SurfaceCodeResult}`.
- `src/vqe.rs` — Variational Quantum Eigensolver: hardware-efficient ansatz, parameter-shift gradient, ground-state energy search (includes `h2_hamiltonian()`).
- `src/grover.rs` — Grover amplitude amplification, direct state-vector oracle access.
- `src/qaoa.rs` — QAOA for MaxCut: parameterized phase-separation + mixing layers.
- `src/surface_code.rs` — distance-3 surface code, stabilizer cycles, noise injection, syndrome decoding.

## Tests

- `tests/test_algorithms.rs` — cross-algorithm correctness tests.

## Dev-deps

`criterion`, `proptest`, `approx` (no benches/ folder configured here).

## Related crates

- `crates/ruqu-core` — quantum-state simulation engine.
