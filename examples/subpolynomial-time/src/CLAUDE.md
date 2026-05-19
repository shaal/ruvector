# subpolynomial-time/src

## Files

- `main.rs` — demo orchestrator. Imports `mod fusion;` and walks
  seven scenarios (basic / dynamic / exact-vs-approx / monitoring /
  resilience / scaling / fusion + brittleness).
- `fusion/` — local Vector-Graph Fusion module used in scenario 7.

## Related

- `../Cargo.toml` — `monitoring` feature on `ruvector-mincut`
- `../../../crates/ruvector-mincut/`
