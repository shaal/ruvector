# ruvix-types/benches

Microbenches for the kernel interface types.

## Files

- `serialization.rs` — round-trip cost of type encoding (used by the queue/IPC layer).
- `type_construction.rs` — construction cost for hot-path types (`CapHandle`, `ProofToken`, etc.).
