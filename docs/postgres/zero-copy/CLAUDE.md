# docs/postgres/zero-copy/

Zero-copy operator implementation details for the ruvector Postgres extension - how distance/score operators are evaluated without copying vector data out of Postgres pages.

## Files

- `ZERO_COPY_IMPLEMENTATION.md` - implementation walkthrough.
- `ZERO_COPY_OPERATORS_SUMMARY.md` - summary of zero-copy operators.
- `zero-copy-operators.md` - operator catalog.
- `examples.rs` - example Rust code for zero-copy operators.

## Related

- `../postgres-zero-copy-memory.md` - parent design doc.
- `../v2/03-index-access-methods.md` - index AMs that use these operators.
