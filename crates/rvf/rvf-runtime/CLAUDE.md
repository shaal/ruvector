# rvf-runtime

User-facing RuVector Format runtime. `RvfStore` is the primary API for creating, opening, querying, and managing RVF vector stores. Ties together the segment model, manifest system, HNSW indexing, quantization, and compaction. Append-only writes; progressive boot; single-writer / multi-reader (advisory lock); background compaction.

## Layout

- `Cargo.toml` — features: `default = ["std"]`, `std`, `wasm`, `qr`, `ed25519` (forwards to `rvf-types/ed25519`). Deps: `rvf-types` (`std`). Dev: `tempfile`.
- `src/lib.rs` — top-level docs + module decls. Lots of modules.
- `src/store.rs` — `RvfStore` primary type.
- `src/options.rs` — `RvfOptions`, `DistanceMetric`, `MetadataEntry`, `MetadataValue`, `QueryOptions`.
- `src/read_path.rs` / `src/write_path.rs` — query / ingest flows.
- `src/compaction.rs` / `src/compress.rs` — dead-space reclamation and segment compression.
- `src/cow.rs` / `src/cow_map.rs` / `src/cow_compact.rs` — RVCOW (ADR-031) copy-on-write branching.
- `src/agi_container.rs` / `src/agi_authority.rs` / `src/agi_coherence.rs` — AGI Cognitive Container system (ADR-036).
- `src/qr_encode.rs` / `src/qr_seed.rs` / `src/seed_crypto.rs` — QR Cognitive Seed bootstrap (ADR-033).
- `src/witness.rs` — WITNESS_SEG integration with `rvf-crypto`.
- `src/locking.rs` — advisory single-writer lock.
- `src/membership.rs`, `src/filter.rs`, `src/deletion.rs`, `src/status.rs`, `src/options.rs` — auxiliary surfaces.
- `src/adversarial.rs` / `src/dos.rs` / `src/safety_net.rs` — hardening against malicious inputs.
- `src/ffi.rs` — C-style FFI bridge.
- `examples/` — runnable demos (`qr_seed_bootstrap`, `qr_seed_encode`, `capability_report`).
- `tests/` — integration tests (`adr033_integration`, `agi_e2e`, `qr_seed_e2e`, `witness_e2e`).

## Public API

`RvfStore` (open/create/insert/query/delete/compact), `RvfOptions`, `QueryOptions`, `DistanceMetric`, filter DSL, AGI container builder, QR seed encoder.

## Related

- `../rvf-types`, `../rvf-wire`, `../rvf-manifest`, `../rvf-index`, `../rvf-quant`, `../rvf-crypto` — building blocks
- `../rvf-server`, `../rvf-node`, `../rvf-cli`, `../rvf-launch`, `../rvf-import` — surfaces over `RvfStore`
- Every adapter under `../rvf-adapters/`
