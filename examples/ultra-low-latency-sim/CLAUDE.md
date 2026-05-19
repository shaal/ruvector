# ultra-low-latency-sim

Meta-simulation engine demonstrating "quadrillion simulations/second"
on CPU-only by stacking five techniques: bit-parallel binary state
(`u64` = 64 sims), SIMD vectorization (NEON/AVX), hierarchical
batching, closed-form solutions, and cache-resident LUTs. Includes
optional Ed25519 verification. Working benchmark binary; standalone
Cargo workspace.

## Important files

- `Cargo.toml` — standalone `[workspace]`; package
  `ultra-low-latency-sim`. Deps: `rayon`, `rand`, `rand_xoshiro`,
  `ed25519-dalek`, `sha2`, `hex`. Target-cfg notes for aarch64 NEON
  and x86_64 AVX2/AVX-512 (runtime-detected).
- `Cargo.lock` — pinned.
- `src/lib.rs` — public surface.
- `src/main.rs` — benchmark driver with `BenchConfig` (verbosity,
  Ed25519 verification toggle).

## Run

```bash
cd examples/ultra-low-latency-sim
cargo run --release
# Verbose + verification:
cargo run --release -- --verbose --verify
```

## Tech stack

- Pure Rust, `rayon` for thread-level parallelism, runtime SIMD dispatch
- Ed25519 (`ed25519-dalek`) + SHA-256 for optional verification path

## Related

- `../exo-ai-2025/research/05-memory-mapped-neural-fields/` — related
  petabyte-scale sim demo
- `../exo-ai-2025/research/01-neuromorphic-spiking/` — bit-parallel
  technique applied to SNNs
