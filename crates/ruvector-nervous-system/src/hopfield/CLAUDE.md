# ruvector-nervous-system/src/hopfield

Hopfield associative-memory network. See `HOPFIELD.md` (crate root) and `examples/hopfield_demo.rs`.

## Files

- `mod.rs` — façade.
- `network.rs` — `HopfieldNetwork` (sync/async update rules).
- `capacity.rs` — capacity analysis helpers (≈0.14·n classical bound).
- `retrieval.rs` — pattern retrieval (energy minimization).
- `tests.rs` — module-level unit tests.
