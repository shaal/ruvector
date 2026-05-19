# rvlite/src

Source for the RvLite WASM vector database with SQL / SPARQL / Cypher.

## Files
- `lib.rs` - top-level `#[wasm_bindgen]` API: `RvLite` JS class,
  `RvLiteConfig`, init/start hook. Module wiring (`cypher`, `sparql`,
  `sql`, `storage`).
- `lib_sql.rs` - additional SQL-focused entry points kept alongside the
  main `lib.rs`.

## Submodules
- `cypher/` - Cypher query engine for the embedded property graph.
- `sparql/` - SPARQL query engine for the embedded triple store.
- `sql/` - SQL engine with pgvector-style vector operators.
- `storage/` - state types, id maps, IndexedDB persistence,
  epoch / writer-lease concurrency control.
