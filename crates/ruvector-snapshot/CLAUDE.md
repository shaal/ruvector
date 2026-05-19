# ruvector-snapshot

Point-in-time snapshots and backup for RuVector collections. Provides compression, SHA-256 checksums, multiple storage backends, and async I/O.

## Important files

- `Cargo.toml` — deps: `ruvector-core`, `serde`, `bincode` (`serde` feature), `flate2`, `sha2`, `tokio` (fs/io-util), `async-trait`, `uuid`, `chrono`.
- `src/lib.rs` — re-exports `SnapshotError`/`Result`, `SnapshotManager`, `Snapshot`, `SnapshotData`, `SnapshotMetadata`, `VectorRecord`, `LocalStorage`, `SnapshotStorage`.

## Module map (src/)

- `manager.rs` — `SnapshotManager`: create / list / restore snapshots.
- `snapshot.rs` — `Snapshot`, `SnapshotData`, `SnapshotMetadata`, `VectorRecord` types and serialization.
- `storage.rs` — `SnapshotStorage` trait + `LocalStorage` filesystem backend (async).
- `error.rs` — `SnapshotError`, `Result`.

## Public API surface

`SnapshotManager`, `Snapshot`, `SnapshotData`, `SnapshotMetadata`, `VectorRecord`, `SnapshotStorage`, `LocalStorage`, `SnapshotError`, `Result`.

## Tests

Inline `#[cfg(test)] mod tests` in `lib.rs` (re-export accessibility checks); deeper testing is integration-side.

## Related

- `crates/ruvector-core` — vector data source.
- `crates/ruvector-raft` — uses snapshots for `InstallSnapshot` RPCs.
