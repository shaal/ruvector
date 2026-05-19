# ruvector-bench/src

Crate root for shared benchmark library code.

## Files

- `lib.rs` — `BenchmarkResult` struct (per-test record) and helpers for synthetic dataset generation (uniform / normal), percentile computation, and JSON serialisation. Used by every binary in `bin/`.
- `bin/` — six benchmark executables, see `bin/CLAUDE.md`.
