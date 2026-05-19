# ruvector-mincut/src/jtree

Dynamic hierarchical J-tree decomposition — the structural backbone of the subpolynomial algorithm (see `docs/adr/ADR-002-dynamic-hierarchical-jtree-decomposition.md`).

## Files

- `mod.rs` — `JTree` façade.
- `coordinator.rs` — coordinates J-tree level updates across the hierarchy.
- `hierarchy.rs` — multi-level hierarchy management.
- `level.rs` — single-level J-tree representation.
- `sparsifier.rs` — sparsifier instance used inside the J-tree.
