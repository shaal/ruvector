# ruvector-graph-transformer-node/src

NAPI bindings plus an embedded graph transformer engine.

## Files
- `lib.rs` - `#[napi]` `GraphTransformer` class binding; delegates to
  `CoreGraphTransformer` from `transformer.rs`. Exposes constructor,
  `version()`, and pipeline / verified-training surface to Node.
- `transformer.rs` - self-contained engine: `CoreGraphTransformer`, `Edge`,
  `PipelineStage`. Kept here so the NAPI surface is decoupled from the
  upstream `ruvector-graph-transformer` crate's API churn.
