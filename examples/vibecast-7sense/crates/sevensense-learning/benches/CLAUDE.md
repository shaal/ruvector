# sevensense-learning/benches

Criterion benchmarks for the learning bounded context.

## Files
- `gnn_benchmark.rs` - Measures GNN forward (and where applicable backward) pass throughput across model types (GCN, GraphSAGE, GAT) and graph sizes.

## Run
```
cargo bench -p sevensense-learning
```

## Related
- Source: `../src/infrastructure/gnn_model.rs`.
