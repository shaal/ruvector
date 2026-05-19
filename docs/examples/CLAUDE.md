# docs/examples/

A small collection of example artifacts that live under `docs/` (the main runnable examples live at the repo root `examples/`). Audience: users looking for code-level usage samples for specific subsystems.

## Files

- `btsp_usage.rs` - usage example for the BTSP (binary-tree segment partition) module.
- `sparsevec_examples.sql` - SQL examples for the pgvector-compatible `sparsevec` type.
- `monitoring_example.md` - example monitoring/observability configuration.

## Subdirs

- `musica/` - a substantial Rust example crate (Musica): audio source separation, hearing-aid, transcription, and visualization built on ruvector. Contains a full Cargo project with `src/`, `wasm/`, `scripts/`, `test_audio/`.

## Related

- `../guides/` - tutorials that walk through examples.
- `../sql/` - additional SQL examples.
- `../postgres/zero-copy/examples.rs` - postgres-specific example code.
