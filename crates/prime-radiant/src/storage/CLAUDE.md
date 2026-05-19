# prime-radiant/src/storage

Pluggable storage backends for witnesses, lineage, and substrate snapshots.

## Files

- `mod.rs` — backend trait + module entry.
- `memory.rs` — in-memory backend (default for tests).
- `file.rs` — append-only file backend.
- `postgres.rs` — Postgres-backed durable store.
