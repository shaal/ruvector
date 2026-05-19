Mirror copy of the vendored `hnsw_rs` v0.3.3 crate. See `../../../patches/hnsw_rs/CLAUDE.md` for the authoritative description - the workspace `Cargo.toml` references the top-level `patches/hnsw_rs/`, not this directory.

Layout matches the upstream:
- `Cargo.toml`, `Changes.md`, `LICENSE-*`
- `src/` - library sources.
- `examples/` - ANN demos.
- `tests/` - integration tests.

This mirror exists for patch-generation tooling under `../../`. Prefer editing the canonical copy at `../../../patches/hnsw_rs/`.
