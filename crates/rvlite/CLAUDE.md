# rvlite

Standalone vector database with SQL (pgvector-compatible), SPARQL, and
Cypher - powered by RuVector WASM. Browser-runnable with IndexedDB
persistence. Ships both as a Rust `rlib` and as a `cdylib` for WASM.

## Important files
- `Cargo.toml` - `crate-type = ["cdylib", "rlib"]`. Deps: `ruvector-core`
  (memory-only), optional `rvf-runtime` / `rvf-types` (RVF backend),
  `wasm-bindgen` family + IndexedDB-heavy `web-sys`, `serde`, `thiserror`.
- `build.rs` - small build helper.
- `src/lib.rs` - top-level `RvLite` JS class + `RvLiteConfig`.
- `src/lib_sql.rs` - alternate / additional SQL surface kept beside
  `lib.rs`.
- `src/cypher/` - Cypher query engine (lexer, parser, AST, executor,
  graph store).
- `src/sparql/` - SPARQL query engine (parser, AST, executor, triple
  store).
- `src/sql/` - SQL engine (parser, AST, executor) with pgvector-style
  syntax.
- `src/storage/` - persistence layer: IndexedDB, id maps, epoch,
  writer-lease.
- `docs/` - design docs and architecture reviews.
- `examples/` - browser demo + full dashboard.
- `tests/` - Rust + WASM integration tests.

## Public API surface
- `RvLite` (JS class) - construct with `RvLiteConfig(dim)`. Methods:
  `insert(vec, metadata)`, `search(query, k)`, `cypher(...)`,
  `add_triple(s, p, o)`, `sparql(...)`, `save()` / `load()` (IndexedDB).
- `GraphState`, `RvLiteState`, `TripleStoreState`, `VectorState` -
  serializable engine state (from `storage`).

## Related
- `../ruvector-core`, `../ruvector-wasm`, `../ruvector-graph-wasm`,
  `../ruvector-gnn-wasm` (the latter planned per Cargo.toml comments).
- Optional `../rvf/rvf-runtime` and `../rvf/rvf-types` for the RVF backend.
