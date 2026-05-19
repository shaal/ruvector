# ruvector-rulake

ruLake — vector-native federation intermediary over heterogeneous backends (ADR-155). A `BackendAdapter` trait, a RaBitQ-compressed cache, and a router that fans queries out across backends and merges top-k by score. Witness digest binds the `table.rulake.json` bundle (uses `sha3` to match `../rvf/rvf-crypto`).

## Layout

- `Cargo.toml` — has a `[[bin]] rulake-demo`. Deps: `ruvector-rabitq`, `serde`/`serde_json`, `thiserror`, `sha3`, `hex`, `rand`/`rand_distr`, `rayon`.
- `BENCHMARK.md` — reproducible benchmark numbers from `bin/rulake-demo`.
- `src/lib.rs` — public API; module decls and ADR-155 overview.
- `src/backend.rs` — `BackendAdapter` trait (`id`, `list_collections`, `pull_vectors`, `generation`).
- `src/fs_backend.rs` — `LocalBackend` in-memory adapter for tests/demos.
- `src/cache.rs` — `VectorCache` wrapping `RabitqPlusIndex`, tracking per-collection generation for staleness.
- `src/lake.rs` — `RuLake` entry point: register backends, run `search` / `search_one`, warm-from-disk.
- `src/bundle.rs` — `table.rulake.json` sidecar serialisation + sha3 witness.
- `src/error.rs` — crate error type.
- `src/bin/rulake-demo.rs` — benchmark harness (see `src/bin/CLAUDE.md`).
- `examples/sidecar_daemon.rs` — long-running publisher/reader cache-coherence demo via `refresh_from_bundle_dir`.
- `examples/warm_restart.rs` — save → ship → warm-restart end-to-end demo.
- `tests/federation_smoke.rs` — federation acceptance gates (top-k parity vs direct RabitQ, cache coherence, federated merge, cache-hit speedup).

## Public API

`BackendAdapter`, `LocalBackend`, `VectorCache`, `RuLake`, bundle/witness helpers.

## Related

- `../ruvector-rabitq` — underlying compressed index
- `../rvf/rvf-crypto` — pinned `sha3` version for witness hashing
- `docs/research/ruLake/` — ADR-155 and implementation plan
