# ruvector-rulake/src

Source for ruLake.

## Files

- `lib.rs` — public API docs and module decls; describes the M1 (MVP) scope.
- `backend.rs` — `BackendAdapter` trait (id, list_collections, pull_vectors, generation token for staleness).
- `fs_backend.rs` — `LocalBackend` in-memory adapter, used in tests/demos.
- `cache.rs` — `VectorCache` wrapping `ruvector_rabitq::RabitqPlusIndex`; tracks per-collection `generation` to detect cache staleness.
- `lake.rs` — `RuLake` public type: register backends, fan-out search via `rayon`, merge top-k by score, warm-from-disk.
- `bundle.rs` — `table.rulake.json` bundle structure + sha3 witness digest.
- `error.rs` — `Error` enum via `thiserror`.
- `bin/rulake-demo.rs` — reproducible benchmark harness backing `BENCHMARK.md`.
