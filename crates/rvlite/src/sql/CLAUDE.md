# rvlite/src/sql

SQL engine with pgvector-compatible syntax for vector columns and operators.

## Files
- `mod.rs` - public API + module wiring.
- `parser.rs` - SQL parser (subset; tuned for pgvector-style queries).
- `ast.rs` - SQL AST types.
- `executor.rs` - AST -> table / vector-index operations; ties into
  `../storage` for persistence.
- `tests.rs` - SQL-side unit tests.
