# ruvector-attention/src/pde_attention

PDE-style attention: treat attention as a diffusion/Laplacian operator on the token graph.

## Files

- `mod.rs` — module entry.
- `diffusion.rs` — explicit / implicit diffusion attention step.
- `laplacian.rs` — Laplacian assembly used by `diffusion.rs`.
