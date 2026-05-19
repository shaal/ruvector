# ruvector-postgres/src/graph/sparql

W3C-compliant SPARQL 1.1 query support for RDF data with a PostgreSQL storage backend plus vector similarity extensions.

Features: SPARQL 1.1 Query (SELECT, CONSTRUCT, ASK, DESCRIBE), Update (INSERT, DELETE, LOAD, CLEAR), RDF triple store with SPO/POS/OSP indexing, property paths (sequence, alternative, inverse, transitive).

## Files

- `mod.rs` — Module entry + re-exports.
- `parser.rs` — SPARQL 1.1 parser.
- `ast.rs` — AST types.
- `executor.rs` — Query executor against the triple store.
- `functions.rs` — Built-in SPARQL functions.
- `triple_store.rs` — RDF triple store with SPO/POS/OSP indexes.
- `results.rs` — Result-set formatting (SELECT/CONSTRUCT/ASK).
