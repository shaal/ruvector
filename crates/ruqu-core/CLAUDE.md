# ruqu-core

Pure-Rust quantum execution intelligence engine for the RuVector stack. Provides state-vector simulation (up to ~32 qubits), stabilizer (millions), Clifford+T, and tensor-network backends with automatic backend routing, noise modelling, error mitigation, hardware abstraction, transpilation, and cryptographic witness logging.

## Features

- `default = ["std"]`
- `parallel` — enables `rayon`
- `simd` — opt-in SIMD kernels

## Layout

- `Cargo.toml` — workspace-tracked. `criterion`, `proptest`, `approx` dev-deps.
- `src/` — large module collection (~30 files); see `src/CLAUDE.md`.
- `benches/quantum_sim.rs` — criterion benchmarks (registered as `[[bench]] name = "quantum_sim"`).
- `tests/` — unit-level integration tests (`test_gates`, `test_simulator`, `test_state`, `test_types`).

## High-level module groups

- Core simulation: `backend`, `circuit`, `gate`, `simulator`, `state`, `stabilizer`, `tensor_network`, `types`, `mixed_precision`, `optimizer`, `simd`.
- Scientific instrument (ADR-QE-015): `confidence`, `hardware`, `mitigation`, `noise`, `qasm`, `replay`, `transpiler`, `verification`, `witness`.
- SOTA differentiation: `clifford_t`, `decomposition`, `pipeline`, `planner`.
- QEC control plane: `control_theory`, `decoder`, `qec_scheduler`, `subpoly_decoder`.
- Benchmark & proof suite: `benchmark`.

## Public API

`use ruqu_core::prelude::*;` exposes `QuantumCircuit`, `Simulator`, `QuantumState`, gate constructors (`circuit.h(0).cnot(0, 1)`), backend trait, noise models, and witness logging.

## Related

- `crates/ruvector-math` — supporting linear algebra / spectral / tensor-network primitives.
- `crates/ruvector-verified` and `ruvector-verified-wasm` — proof-carrying validators for results emitted here.
