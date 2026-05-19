# ruvector-mincut-gated-transformer/src/attention

Pluggable attention modules selectable via crate features.

## Files

- `mod.rs` — common trait + module re-exports.
- `window.rs` — sliding-window attention (default via `sliding_window` feature).
- `linear.rs` — Performer-style linear attention (`linear_attention` feature).
- `spike_driven.rs` — spike-driven attention (`spike_attention` feature).

The standalone files `../flash_attention.rs`, `../sparse_attention.rs`, `../spike.rs`, `../spectral.rs` provide additional attention variants outside this folder.
