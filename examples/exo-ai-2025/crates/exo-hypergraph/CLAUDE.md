# exo-hypergraph

Hypergraph substrate for higher-order relational reasoning. Provides
hyperedge containers, sheaf-cohomology helpers, and sparse topological
data analysis (TDA) routines on top of exo-core.

## Files

- `Cargo.toml` — depends on `exo-core`, serde, thiserror, uuid,
  dashmap, petgraph.
- `src/lib.rs` — re-exports.
- `src/hyperedge.rs` — hyperedge / hypergraph types.
- `src/topology.rs` — topology helpers.
- `src/sheaf.rs` — sheaf cohomology computations.
- `src/sparse_tda.rs` — sparse persistent-homology / TDA primitives.
- `tests/hypergraph_test.rs` — coverage for the above.

## Build / Test

```bash
cargo build -p exo-hypergraph
cargo test  -p exo-hypergraph
```

## Related

- `../../research/04-sparse-persistent-homology/` — standalone
  research version
- `../../benches/hypergraph_bench.rs`
