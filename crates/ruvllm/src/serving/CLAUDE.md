# ruvllm/src/serving

Continuous batching serving engine. Dynamic batching of prefill + decode
with a scheduler that moves requests through Pending -> Running -> Completed
queues. Backs the HTTP server in `ruvllm-cli serve`.

## Files
- `mod.rs` - public API + architecture diagram.
- `engine.rs` - `ServingEngine` (the public driver).
- `scheduler.rs` - request scheduler (batching, prioritization).
- `request.rs` - request types (inputs, sampling params, callbacks).
- `batch.rs` - batch packing / unpacking helpers.
- `kv_cache_manager.rs` - KV-cache lifecycle bridge to
  `../paged_attention.rs` and `../kv_cache.rs`.
