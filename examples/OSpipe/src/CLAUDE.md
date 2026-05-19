# OSpipe / src

Rust source tree for the OSpipe crate. Contains the library entry point, cross-cutting modules, and one subdirectory per pipeline stage.

## Top-level files
- `lib.rs` - crate root, wires the submodules together.
- `config.rs` - configuration types and loading.
- `error.rs` - shared `Error`/`Result` types (`thiserror`).
- `persistence.rs` - on-disk persistence helpers.
- `safety.rs` - safety / containment guards used by ingestion + search.

## Subdirectories
- `bin/` - executable entry points (`ospipe-server`).
- `capture/` - frame capture surface that adapts Screenpipe-style inputs.
- `pipeline/` - ingestion pipeline plus dedup logic.
- `storage/` - vector store, embedding interface, persistence traits.
- `search/` - hybrid + enhanced search, reranker, router, MMR.
- `graph/` - entity extraction over captured content.
- `learning/` - learning loops on top of captured/queried data.
- `quantum/` - experimental quantum-inspired primitives.
- `server/` - Axum HTTP layer.
- `wasm/` - `wasm-bindgen` surface exposed to JS.

## Build
- Native: `cargo build -p ospipe`.
- WASM: `wasm-pack build --target web ../`.
