# ruvector-attention/src

Source root for the attention crate. See `lib.rs` for the canonical module list and re-exports.

## Top-level files

- `lib.rs` — crate doc, module declarations, top-level re-exports.
- `config.rs` — shared attention configs.
- `error.rs` — `AttentionError` + Result alias.
- `traits.rs` — the central `Attention` trait that all kernels implement.
- `utils.rs` — small math/util helpers (softmax, etc.).

## Subdirectories

| Dir | Purpose |
|---|---|
| `attention/` | core kernels: scaled-dot-product, multi-head, flash, MLA, SSM/Mamba, speculative, KV cache |
| `curvature/` | curvature primitives (component quantizer, fused attention, tangent space) |
| `graph/` | graph attention (dual-space, edge-featured, RoPE) |
| `hyperbolic/` | hyperbolic / Poincare / Lorentz / mixed-curvature |
| `info_bottleneck/` | information-bottleneck attention + KL |
| `info_geometry/` | Fisher info + natural gradient |
| `moe/` | Mixture-of-Experts attention |
| `pde_attention/` | PDE diffusion + Laplacian |
| `sdk/` | high-level builder / pipeline / presets |
| `sheaf/` | sheaf attention + early exit + sparse router (ADR-015) |
| `sparse/` | sparse patterns (flash, linear, local+global, mask) |
| `topology/` | topology-gated attention + coherence + policy |
| `training/` | training loop helpers (curriculum, loss, mining, optimizer) |
| `transport/` | optimal-transport variants (sliced wasserstein, centroid OT) |
| `unified_report/` | metric + report aggregation |
