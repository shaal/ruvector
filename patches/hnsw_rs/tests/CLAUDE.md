Integration tests for the vendored `hnsw_rs` crate. Run with `cargo test` in the parent directory.

Files:
- `deallocation_test.rs` - verifies proper drop/cleanup of HNSW structures.
- `filtertest.rs` - exercises the search-time filter trait (`src/filter.rs`).
- `serpar.rs` - serialization and parallel build/query tests.
