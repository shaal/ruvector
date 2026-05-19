# rvlite/src/cypher

Cypher query engine over the embedded property graph (`GraphState` in
`../storage`). Supports a subset of openCypher.

## Files
- `mod.rs` - public API + module wiring.
- `lexer.rs` - tokenizer.
- `parser.rs` - token stream -> AST.
- `ast.rs` - Cypher AST types (`Match`, `Create`, patterns, expressions).
- `executor.rs` - AST -> graph-store ops + result rows.
- `graph_store.rs` - in-memory property graph (nodes, relationships,
  labels, props).
