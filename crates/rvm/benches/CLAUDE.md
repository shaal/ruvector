# rvm/benches

Workspace member (`rvm-benches`, `publish = false`) holding criterion benchmarks for the whole RVM stack.

- `Cargo.toml` — declares the package and three `[[bench]]` targets, all `harness = false`. Pulls every RVM crate via `workspace = true`.
- `src/lib.rs` — empty / shared scaffolding for the bench crate.
- `benches/` — criterion sources; see `benches/CLAUDE.md`.

See `../CLAUDE.md`.
