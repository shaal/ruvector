# ruvector-attention

Attention mechanisms for ruvector — geometric, graph, and sparse attention. Implementations include scaled dot-product, multi-head, FlashAttention-3, MLA with KV-cache compression, selective state space (Mamba), speculative decoding, sparse attention, hyperbolic / mixed-curvature, sheaf attention (Coherence-Gated Transformer per ADR-015), MoE, PDE diffusion, and topology-gated attention.

## Layout

- `Cargo.toml` — features: `simd` (default), `wasm`, `napi`, `math` (opt-in `ruvector-math` deps for OT, mixed-curvature, topology-gated), `sheaf` (Coherence-Gated Transformer). Deps: `thiserror`, `rayon`, `serde`, `rand`, optional `napi`/`napi-derive` + `ruvector-math`. Benches via `criterion`; one `bench_runner` binary.
- `src/lib.rs` — re-exports trait `Attention` + `ScaledDotProductAttention` example. Module declarations: `attention, config, error, graph, hyperbolic, info_bottleneck, info_geometry, moe, pde_attention, sdk, sheaf, sparse, topology, training, transport, unified_report`, plus root `traits` and `utils`.
- `src/config.rs`, `src/error.rs`, `src/traits.rs`, `src/utils.rs` — top-level glue.
- `src/curvature/` — curvature primitives (component quantizer, fused attention, tangent space).

## Module groups (under `src/`)

- `attention/` — core attention kernels (scaled-dot-product, multi-head, flash, MLA, SSM/Mamba, speculative decoding, KV cache).
- `graph/` — graph attention (dual-space, edge-featured, RoPE).
- `hyperbolic/` — Poincare / Lorentz / mixed-curvature attention.
- `info_bottleneck/`, `info_geometry/` — bottleneck + Fisher / natural gradient.
- `moe/` — Mixture of Experts attention (router, expert, attention).
- `pde_attention/` — diffusion + Laplacian PDE attention.
- `sheaf/` — sheaf attention + early exit + sparse router (ADR-015).
- `sparse/` — sparse patterns (flash, linear, local+global, mask).
- `topology/` — topology-gated attention + coherence + policy.
- `transport/` — optimal-transport (centroid OT, sliced Wasserstein, cached projections).
- `training/` — curriculum + loss + mining + optimizer for training attention modules.
- `sdk/` — high-level builder / pipeline / presets.
- `unified_report/` — metric + report aggregation across mechanisms.

## Tests / benches / examples

- `benches/attention_bench.rs`, `benches/attention_benchmarks.rs` (also wired as the `bench_runner` bin).
- `examples/hyperbolic_bench.rs` — runnable example.
- `docs/IMPLEMENTATION_SUMMARY.md`, `docs/SDK_GUIDE.md`.

## Related crates

- `crates/ruvector-math` — advanced math primitives (optional dep behind `math` feature).
- `crates/prime-radiant` — consumes topology / MoE / PDE attention via its adapter.
- `crates/ruvector-crv` — depends on this crate for Stage II vectors.
