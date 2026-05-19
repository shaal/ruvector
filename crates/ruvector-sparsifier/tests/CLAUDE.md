# ruvector-sparsifier/tests

Integration tests for the sparsifier.

## Files

- `integration_tests.rs` — end-to-end tests covering: building H from a SparseGraph, dynamic insert/delete edges, audit drift
  detection, and compression-ratio sanity.

Run: `cargo test -p ruvector-sparsifier`.
