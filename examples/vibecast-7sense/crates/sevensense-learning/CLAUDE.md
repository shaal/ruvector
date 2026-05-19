# sevensense-learning

Graph Neural Network (GNN)-based learning and embedding refinement for 7sense: GCN / GraphSAGE / GAT models, contrastive learning (InfoNCE, triplet), Elastic Weight Consolidation (EWC) for continual learning, attention mechanisms.

## Files
- `Cargo.toml` - Depends on `sevensense-core`, `sevensense-vector`, async + math libs.
- `src/lib.rs` - Crate root and architecture overview; re-exports `LearningService`, `LearningConfig`, `GnnModelType`.
- `src/loss.rs` - Contrastive losses (InfoNCE, triplet).
- `src/ewc.rs` - Elastic Weight Consolidation for continual learning.
- `src/domain/` - Domain entities and repository traits.
- `src/application/` - `LearningService` and friends.
- `src/infrastructure/` - GNN model implementations, attention.
- `benches/gnn_benchmark.rs` - Criterion benchmark for GNN forward/backward.

## Build / bench
```
cargo build -p sevensense-learning
cargo bench -p sevensense-learning
```

## Related
- Consumed by `sevensense-api`.
- Uses `sevensense-vector` for graph/edge storage.
