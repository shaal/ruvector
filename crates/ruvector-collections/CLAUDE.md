# ruvector-collections

Multi-collection management for ruvector vector databases — group vectors into collections, manage aliases, gather stats — plus the workspace's shared primality utility (ADR-151 / PIAL): deterministic Miller-Rabin + tabled fast paths for prime moduli used by ruvector-graph, micro-hnsw-wasm, sparsifier, attn-mincut, and pi-brain.

## Layout

- `Cargo.toml` — deps: `ruvector-core`, `dashmap`, `parking_lot`, `uuid`, `bincode`, `chrono`. Feature `unstable-u128` opts into probabilistic Miller-Rabin for u128 (kept out of WASM default per ADR-151).
- `build.rs` — generates lookup tables (likely prime / pseudoprime tables) at build time.
- `src/lib.rs` — module declarations + crate docs (warn(missing_docs)). Example shows `CollectionManager` + `CollectionConfig` usage.
- `src/manager.rs` — `CollectionManager` (create, alias, get; DashMap-backed for concurrency).
- `src/collection.rs` — `Collection` + `CollectionConfig` (dimensions, distance metric, HNSW config, quantization, on-disk payload).
- `src/error.rs` — crate errors.
- `src/primality.rs` — public Miller-Rabin API (deterministic for u64).
- `src/primality_kernel.rs` — inner kernel + tabled fast paths.

## Tests / benches

- `benches/primality.rs` — criterion bench for primality kernel.
- `tests/primality_pseudoprimes.rs` — known pseudoprime corner cases.
- `tests/table_cross_check.rs` — generated-table consistency check.

## Related crates

- `crates/ruvector-core` — base vector DB types (`HnswConfig`, `DistanceMetric`).
- Primality consumers: `ruvector-graph`, `micro-hnsw-wasm`, `ruvector-sparsifier`, `attn-mincut`, `pi-brain`.
