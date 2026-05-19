# 03-time-crystal-cognition

Standalone research crate: cognitive time crystals — discrete time
translation symmetry breaking applied to working memory. Models a
Floquet driven system whose periodic response stores information.

## Files

- `Cargo.toml` — standalone `[workspace]`; package
  `time-crystal-cognition`. Deps: `ndarray`, `rand`, `rustfft`. Bench
  `time_crystal_bench`.
- `RESEARCH.md`, `BREAKTHROUGH_HYPOTHESIS.md`, `EXECUTIVE_SUMMARY.md`,
  `mathematical_framework.md` — theory + framework.
- `Cargo.lock` — pinned.
- `src/lib.rs` — public surface.
- `src/discrete_time_crystal.rs` — DTC dynamics.
- `src/floquet_cognition.rs` — Floquet-driven cognitive layer.
- `src/temporal_memory.rs` — temporal memory store atop DTC.
- `src/simd_optimizations.rs` — SIMD kernels.
- `benches/time_crystal_bench.rs` — Criterion suite.

## Build / Bench

```bash
cd examples/exo-ai-2025/research/03-time-crystal-cognition
cargo build --release
cargo bench
```

## Related

- `../../crates/exo-temporal/` — temporal memory in the production crate
- `../../crates/exo-exotic/src/experiments/time_crystal_cognition.rs`
