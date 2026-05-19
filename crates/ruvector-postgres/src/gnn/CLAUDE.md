# ruvector-postgres/src/gnn

Graph Neural Network module — GNN-based embeddings and graph-aware vector operations exposed as PostgreSQL operators.

## Files

- `mod.rs` — Module entry; re-exports `operators::*`.
- `gcn.rs` — Graph Convolutional Network layer.
- `graphsage.rs` — GraphSAGE layer.
- `message_passing.rs` — Generic message-passing primitives.
- `aggregators.rs` — Sum/Mean/Max/LSTM aggregators.
- `operators.rs` — pgrx SQL operator wrappers.
