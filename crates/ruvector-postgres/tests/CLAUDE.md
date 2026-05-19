# ruvector-postgres/tests

SQL integration tests for the extension.

## Files

- `hnsw_index_tests.sql` — HNSW (`ruhnsw`) access-method coverage.
- `ivfflat_am_test.sql` — IVFFlat (`ruivfflat`) access-method coverage.

Run via `cargo pgrx test` (uses `pgrx-tests` per the `pg_test` feature). Rust-level inline tests live alongside their modules in `../src/**/tests.rs`.
