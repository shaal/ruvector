# 11-conscious-language-interface

Standalone research crate: integration of ruvLLM + neuromorphic spiking
+ ruvector for conscious AI exposed through a natural-language
interface. Includes spike <-> embedding bridges, qualia memory, novel
learning rules, and intelligence metrics.

## Files

- `Cargo.toml` — standalone `[workspace]`; package
  `conscious-language-interface`. No runtime deps; dev-dep `criterion`;
  optional `simd` feature. Bench `consciousness_bench`.
- `ARCHITECTURE.md`, `BENCHMARK_RESULTS.md` — design + numbers.
- `Cargo.lock` — pinned.
- `src/lib.rs` — re-exports.
- `src/consciousness_router.rs` — routes language queries through the
  consciousness layer.
- `src/intelligence_metrics.rs` — proposed intelligence metrics.
- `src/qualia_memory.rs` — qualia-based memory store.
- `src/spike_embedding_bridge.rs` — converts between spike trains and
  embeddings.
- `src/novel_learning.rs`, `src/advanced_learning.rs` — learning rules.
- `benches/consciousness_bench.rs` — Criterion suite.

## Build / Bench

```bash
cd examples/exo-ai-2025/research/11-conscious-language-interface
cargo build --release
cargo bench
# Optional SIMD feature:
cargo build --release --features simd
```

## Related

- `../01-neuromorphic-spiking/` — spiking primitives
- `../../report/INTELLIGENCE_METRICS.md`
