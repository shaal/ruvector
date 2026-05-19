# ruvector-snapshot/src

Snapshot/backup source.

## Files

- `lib.rs` — module decls + public re-exports + inline export-check tests.
- `manager.rs` — `SnapshotManager` orchestrates create / list / restore over a `SnapshotStorage`.
- `snapshot.rs` — `Snapshot`, `SnapshotData`, `SnapshotMetadata`, `VectorRecord` types (bincode/serde + flate2 compression + sha2 checksums).
- `storage.rs` — `SnapshotStorage` async trait + `LocalStorage` (tokio fs) backend.
- `error.rs` — `SnapshotError` and `Result<T>` alias.
