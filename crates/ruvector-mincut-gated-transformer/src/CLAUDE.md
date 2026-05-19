# ruvector-mincut-gated-transformer/src

Implementation of the mincut-gated transformer. Many modules are feature-gated (see `Cargo.toml`).

## Top-level

- `lib.rs` — module declarations, public re-exports, doc-comment overview.
- `model.rs` — top-level transformer model.
- `state.rs` — model state (KV cache handles, RoPE state, gate state).
- `config.rs` — `ModelConfig` and related configs.
- `packets.rs` — token-packet I/O abstraction.
- `error.rs` — crate error enum.

## Attention

- `attention/` — pluggable attention modules (`linear.rs`, `window.rs`, `spike_driven.rs`).
- `flash_attention.rs` — Flash Attention.
- `sparse_attention.rs` — mincut-aware sparse attention (`sparse_attention` feature).
- `spike.rs` — spike-driven attention (`spike_attention` feature).
- `spectral.rs` — spectral position encoding (`spectral_pe` feature).
- `rope.rs` — RoPE positional encoding.

## Gates / scheduling

- `gate.rs` — min-cut coherence gate.
- `energy_gate.rs` — energy-based gate (`energy_gate` feature).
- `mod_routing.rs` — Mixture-of-Depths router.
- `early_exit.rs` — layer-skip early exit.
- `speculative.rs` — speculative decoding.

## Layers / kernels

- `ffn.rs` — feed-forward block.
- `kernel/` — INT4/Q15 GEMM, RMSNorm.
- `q15.rs` — Q15 fixed-point helpers (`fixed_point_softmax`).
- `mamba.rs` — Mamba SSM block.
- `kv_cache/` — KV-cache subsystem.

## Perf / observability

- `arena.rs` — bump arena for allocation-free hot path.
- `trace.rs` — observability under the `trace` feature.
