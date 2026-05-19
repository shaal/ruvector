# ruvector-mincut-gated-transformer/src/kv_cache

KV-cache subsystem — multi-tier quantised cache with metrics and eviction policies.

## Files

- `mod.rs` — façade and re-exports.
- `manager.rs` — top-level cache manager.
- `tier.rs` — tiered storage (hot / warm / cold).
- `hot_buffer.rs` — full-precision hot buffer for recent tokens.
- `quantized_store.rs` — quantised backing store for older tokens.
- `kivi.rs` — KIVI quantisation scheme.
- `kvquant.rs` — KVQuant quantisation scheme.
- `squat.rs` — SquAt (squeeze-and-attend) variant.
- `policy.rs` — eviction / tier-transition policies.
- `metrics.rs` — cache metrics (hit-rate, residency).
- `legacy.rs` — legacy backend kept for compatibility.
