# rvlite/src/sparql

SPARQL query engine over the embedded triple store.

## Files
- `mod.rs` - public API + module wiring.
- `parser.rs` - SPARQL parser (subset).
- `ast.rs` - SPARQL AST types (`Select`, basic graph patterns,
  filters, etc.).
- `executor.rs` - AST -> triple-store scans + result bindings.
- `triple_store.rs` - in-memory RDF triple store; the `add_triple(...)`
  JS method writes here.
