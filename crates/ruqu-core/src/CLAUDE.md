# ruqu-core/src

Module map for the quantum execution engine. All modules are declared in `lib.rs`.

## Core simulation

- `backend.rs` — backend trait + dispatch for state-vector / stabilizer / Clifford+T / tensor-network.
- `circuit.rs` — `QuantumCircuit` builder and IR.
- `circuit_analyzer.rs` — depth / two-qubit count / entanglement metrics.
- `gate.rs` — gate definitions (`H`, `CNOT`, `T`, parametric rotations, ...).
- `simulator.rs` — top-level `Simulator::run` orchestrator.
- `state.rs` — `QuantumState` representation + probability extraction.
- `stabilizer.rs` — stabilizer-formalism backend (millions of qubits when Clifford-only).
- `tensor_network.rs` — tensor-network backend.
- `types.rs` — shared scalar / amplitude types.
- `mixed_precision.rs` — mixed FP backend kernels.
- `optimizer.rs` — circuit simplification passes.
- `simd.rs` — SIMD kernels for state-vector evolution (feature `simd`).

## Scientific instrument layer (ADR-QE-015)

- `confidence.rs`, `hardware.rs`, `mitigation.rs`, `noise.rs`, `qasm.rs`, `replay.rs`, `transpiler.rs`, `verification.rs`, `witness.rs`.

## SOTA differentiation

- `clifford_t.rs`, `decomposition.rs`, `pipeline.rs`, `planner.rs`.

## QEC control plane

- `control_theory.rs`, `decoder.rs`, `qec_scheduler.rs`, `subpoly_decoder.rs`.

## Misc

- `benchmark.rs` — proof-suite helpers used by both `benches/` and `tests/`.
- `error.rs` — crate error type via `thiserror`.
- `lib.rs` — crate root, re-exports `prelude`.

See `../CLAUDE.md`.
