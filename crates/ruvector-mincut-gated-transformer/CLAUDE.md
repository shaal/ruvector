# ruvector-mincut-gated-transformer

Ultra-low-latency transformer inference engine for continuous systems. Inference is governed by a coherence controller driven by **dynamic minimum-cut signals**, and optionally a spiking scheduler that skips work when nothing meaningful is happening. Combines SOTA optimization techniques (Mixture-of-Depths, Early Exit, Sparse Attention, Energy-Based Transformers, Spike-Driven Inference, Spectral Methods) into one deterministic, allocation-free inference path.

## Important files

- `Cargo.toml` — `rlib`-only. Rich feature surface: `sliding_window` (default), `linear_attention`, `spike_attention`, `spectral_pe`, `sparse_attention`, `energy_gate`, `simd`, `int4`, `fixed_point_softmax`, `rmsnorm`, `trace`, plus `full` aggregate.
- `src/lib.rs` — crate doc enumerating academic foundations and primary design outcomes (deterministic, allocation-free, predictable p99, explainable witnesses).
- `docs/THEORY.md`, `docs/BENCHMARKS.md`, `docs/CITATIONS.bib`, `docs/flash_attention_implementation.md` — theory, measurements, references.
- `examples/` — `flash_attention_demo.rs`, `mamba_example.rs`, `scorer.rs`.
- `benches/` — `gate.rs`, `kernel.rs`, `latency.rs`.

## Module map (src/)

Top-level inference:
- `model.rs` — top-level transformer model assembly.
- `state.rs`, `packets.rs` — model state and packetized I/O.
- `config.rs`, `error.rs`, `lib.rs` — configuration, errors, module wiring.

Attention variants:
- `attention/` — `linear.rs` (Performer), `window.rs` (sliding), `spike_driven.rs`, plus shared `mod.rs`.
- `flash_attention.rs` — Flash Attention implementation.
- `sparse_attention.rs` — mincut-aware sparse attention (MInference 2024).
- `spike.rs` — spike-driven attention (Yao et al.).
- `spectral.rs` — spectral position encoding (Kreuzer et al.).
- `rope.rs` — RoPE positional encoding.

Gates / scheduling:
- `gate.rs` — min-cut coherence gate.
- `energy_gate.rs` — energy-based gate policy (Gladstone et al., 2025).
- `mod_routing.rs` — Mixture-of-Depths routing (Raposo et al., 2024).
- `early_exit.rs` — layer-skip early exit (Elhoushi et al., 2024).
- `speculative.rs` — speculative decoding.

Layers / kernels:
- `ffn.rs` — feed-forward block.
- `kernel/` — `qgemm.rs`, `quant4.rs`, `norm.rs`, `bench_utils.rs` — quantised GEMM, INT4 packing, RMSNorm.
- `q15.rs` — Q15 fixed-point helpers.
- `mamba.rs` — Mamba SSM block.
- `kv_cache/` — KV-cache subsystem (hot-buffer, KiVi, KVQuant, SquAt, policy, tier, manager, metrics, legacy, quantized-store).

Perf / observability:
- `arena.rs` — bump arena (allocation-free hot path).
- `trace.rs` — `trace` feature observability.

## Tests

`tests/` covers determinism (and extended), early exit, energy gate, gate, mod_routing, sparse_attention, spectral, spike_attention, integration, and a generic `verification.rs`.

## Public API surface

Model builder, attention modules, gate types, KV-cache manager — see `src/lib.rs` re-exports.

## Related

- `crates/ruvector-attn-mincut` — the standalone min-cut attention operator.
- `crates/ruvector-mincut` — dynamic min-cut algorithm that produces the gating signal.
