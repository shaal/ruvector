# prime-radiant/src/gpu/shaders

WGSL compute shaders dispatched by `gpu/dispatch.rs`.

## Files

- `types.wgsl` — shared struct + buffer-binding definitions.
- `compute_residuals.wgsl` — per-edge `r_e = rho_u(x_u) - rho_v(x_v)` kernel.
- `compute_energy.wgsl` — energy aggregation `E = sum(w_e * |r_e|^2)` with atomics.
- `sheaf_attention.wgsl` — sheaf-attention forward pass.
- `sparse_mask.wgsl` — sparse-mask application for sheaf attention.
- `token_routing.wgsl` — MoE token routing on GPU.

Shaders are loaded as strings at runtime; keep struct layouts in sync with `kernels.rs` / `buffer.rs`.
