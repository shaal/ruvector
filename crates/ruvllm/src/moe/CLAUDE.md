# ruvllm/src/moe

Mixture-of-Experts components: routing metrics, expert affinity, memory-aware
routing (ADR-092), and SRAM-aware placement.

## Files
- `mod.rs` - public API + design notes.
- `router.rs` - `MemoryAwareRouter` + `RouterConfig` (cache-residency bonus,
  load balancing).
- `affinity.rs` - expert co-activation / affinity tracker.
- `metrics.rs` - routing / paging / latency metrics.
- `precision_allocator.rs` - per-expert precision allocation
  (mixed-precision MoE).
- `sram_mapper.rs` - map experts onto SRAM banks / scratchpad memory.
