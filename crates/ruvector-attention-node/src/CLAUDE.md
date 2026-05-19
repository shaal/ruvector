# ruvector-attention-node/src

NAPI-RS binding source. Each module wraps a slice of `ruvector-attention` for Node.

- `lib.rs` — module roots, re-exports, `version()` `#[napi]` function.
- `attention.rs` — wrappers for `DotProductAttention`, `MultiHeadAttention`,
  `FlashAttention`, `HyperbolicAttention`, `LinearAttention`, `LocalGlobalAttention`,
  `MoEAttention`, plus their config types.
- `training.rs` — optimizers (Adam, AdamW, SGD), losses (InfoNCE, LocalContrastive),
  miners (HardNegative, InBatch), schedulers, regularization.
- `async_ops.rs` — `BatchConfig`, `BatchResult`, `ParallelConfig`, `StreamProcessor`,
  `BenchmarkResult`, plus the `AttentionType` discriminator.
- `graph.rs` — `DualSpaceAttention`, `EdgeFeaturedAttention`, `GraphRoPEAttention`,
  `RoPEConfig`, `DualSpaceConfig`, `EdgeFeaturedConfig`.
