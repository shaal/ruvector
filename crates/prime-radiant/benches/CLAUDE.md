# prime-radiant/benches

Criterion benchmarks covering every major subsystem.

## Files

- `attention_bench.rs` — topology-gated/MoE/PDE attention throughput.
- `coherence_bench.rs`, `coherence_benchmarks.rs` — engine end-to-end residual + energy.
- `energy_bench.rs` — aggregated energy E(S) over substrate.
- `gate_bench.rs` — 4-lane execution gate dispatch latency.
- `gpu_benchmarks.rs` — wgpu pipeline overhead vs CPU.
- `hyperbolic_bench.rs` — Poincare depth + Mobius ops.
- `incremental_bench.rs` — delta updates from `coherence/incremental.rs`.
- `mincut_bench.rs` — partitioning round trip.
- `residual_bench.rs` — restriction-map residual r_e.
- `simd_benchmarks.rs` — SIMD vs scalar inner loops.
- `sona_bench.rs` — sona-tuned threshold convergence.
- `tile_bench.rs` — 256-tile fabric write/sync.

Run with `cargo bench -p prime-radiant`.
