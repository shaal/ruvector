# sevensense-learning/src

Source for the learning bounded context.

## Files
- `lib.rs` - Crate root; documents the DDD layout and re-exports the public API.
- `loss.rs` - Contrastive loss functions (InfoNCE, triplet).
- `ewc.rs` - Elastic Weight Consolidation for continual learning.

## Subdirectories
- `domain/` - Entities and repository traits.
- `application/` - Learning use case services.
- `infrastructure/` - GNN model implementations (GCN/GraphSAGE/GAT) and attention.

## Related
- Parent: `../CLAUDE.md`.
