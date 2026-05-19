# ruqu-core/tests

Black-box integration tests against the public API of `ruqu-core`.

- `test_gates.rs` — gate-by-gate correctness vs analytic expectations.
- `test_simulator.rs` — end-to-end `Simulator::run` paths, includes Bell-state and small algorithms.
- `test_state.rs` — `QuantumState` invariants and probability normalisation.
- `test_types.rs` — scalar / amplitude type behaviour.

See `../CLAUDE.md`.
