# ruvLLM / esp32 / src / federation

Multi-chip federation layer. Lets a cluster of ESP32 devices coordinate on a single inference: sharding, speculative decoding, tensor parallelism, and a small FastGRNN router.

## Important files
- `mod.rs` - module root.
- `coordinator.rs` - federation coordinator that orchestrates child chips.
- `protocol.rs` - wire protocol between chips.
- `pipeline.rs` - pipeline-parallel scheduling.
- `sharding.rs` - per-chip weight/activation shards.
- `tensor_parallel.rs` - tensor-parallel splitting.
- `speculative.rs` - speculative decoding for latency.
- `fastgrnn_router.rs` - FastGRNN router that picks which chip handles a request.
- `massive_scale.rs`, `medium_scale.rs` - scale-specific federation modes.

## Related
- Higher-level demo: `../../examples/federation_demo.rs`. Companion flashable variant: `../../../esp32-flash/src/federation/`.
