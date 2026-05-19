# ultra-low-latency-sim/src

## Files

- `lib.rs` — public surface re-exporting the kernels below.
- `main.rs` — benchmark driver with `BenchConfig` (`enable_verification`,
  `verbose`); orchestrates the five techniques and prints throughput.
- `bit_parallel.rs` — bit-parallel simulation (`u64` = 64 binary states).
- `simd_ops.rs` — NEON / AVX SIMD kernels with runtime dispatch.
- `hierarchical.rs` — hierarchical batching (each op stands in for
  many meta-level outcomes).
- `closed_form.rs` — closed-form analytical replacements for iterative
  loops.
- `verify.rs` — Ed25519 + SHA-256 cryptographic verification path.

## Related

- `../Cargo.toml` — deps + target-cfg
- Sibling research crate using bit-parallel SNNs:
  `../../exo-ai-2025/research/01-neuromorphic-spiking/`
