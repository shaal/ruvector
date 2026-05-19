# ruvector-temporal-tensor/tests

Test suites covering compression correctness, performance, and FFI.

- `integration.rs` — end-to-end push/flush + decompress round-trips.
- `benchmarks.rs` — performance smoke / ratio checks (cargo-test based).
- `property_tests.rs` — round-trip and tier-policy invariants.
- `stress_tests.rs` — long-running stress workloads.
- `persistence_tests.rs` — disk-backed store (feature `persistence`).
- `wasm_ffi_test.rs` — WASM/C FFI surface (feature `ffi`).
