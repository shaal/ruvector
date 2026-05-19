# graph

Standalone Rust files illustrating intended `ruvector-graph` (graph database) API usage. These are documentation-style stubs: most code is commented out pending the public graph API, so they currently print walkthrough placeholders.

## Files

- `basic_graph.rs` - Nodes, properties, relationships, basic CRUD.
- `cypher_queries.rs` - Cypher-like query examples.
- `hybrid_search.rs` - Combined vector + graph queries.
- `distributed_cluster.rs` - Multi-node cluster topology example.

## How to run

There is no `Cargo.toml` here; copy into a project that depends on `ruvector-graph`, or compile ad hoc:

```bash
rustc basic_graph.rs && ./basic_graph
```

## Tech stack

- Rust (plain `.rs` files, no manifest).
- Target consumer: `crates/ruvector-graph` (when the API is exposed).

## Related

- Graph crate: `crates/ruvector-graph`.
- Other graph-flavored demos: `examples/data/framework/examples/dynamic_mincut_demo.rs`, `examples/rvf/examples/causal_atlas.rs`.
