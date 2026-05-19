# ruvector-mincut/src/subpolynomial

The headline subpolynomial-time algorithm: O(n^{o(1)}) amortized update time for cuts up to 2^{O((log n)^{3/4})}. Ties together `jtree/`, `canonical/`, `sparsify/`, and `expander/`.

- `mod.rs` — algorithm driver and public entry points.

See `docs/adr/ADR-002-dynamic-hierarchical-jtree-decomposition.md` for the design and `examples/subpoly_bench.rs` for a runnable demo.
