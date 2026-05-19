# 01-neuromorphic-spiking

Standalone research crate: bit-parallel spiking neural networks
for consciousness computation. Treats each `u64` as 64 simultaneous
spike states for high-throughput SNN simulation.

## Files

- `Cargo.toml` — standalone `[workspace]`; package
  `neuromorphic-spiking`; dep `rand`, dev-dep `criterion`.
- `RESEARCH.md`, `BREAKTHROUGH_HYPOTHESIS.md`, `benchmarks.md` —
  background, hypothesis, and recorded benchmark numbers.
- `Cargo.lock` — pinned for reproducible bench runs.
- `src/lib.rs` — public surface.
- `src/bit_parallel_spikes.rs` — 64-wide bit-parallel spike kernels.
- `src/spiking_consciousness.rs` — Phi-style consciousness measurement
  over spiking activity.
- `benches/spike_benchmark.rs` — Criterion suite.
- `examples/quick_bench.rs` — quick smoke-bench runnable with
  `cargo run --example quick_bench`.

## Build / Run

```bash
cd examples/exo-ai-2025/research/01-neuromorphic-spiking
cargo build --release
cargo bench
cargo run --example quick_bench --release
```

## Related

- `../../crates/exo-exotic/src/experiments/neuromorphic_spiking.rs` —
  exo-substrate-flavored adaptation
- `../../crates/exo-core/src/backends/neuromorphic.rs`
