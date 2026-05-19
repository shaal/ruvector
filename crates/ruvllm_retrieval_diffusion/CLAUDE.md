# ruvllm_retrieval_diffusion

Corpus-agnostic, training-free retrieval LM and masked discrete diffusion built on `ruvllm_sparse_attention`. Generalises the `sparse-mario` example: any small-vocab token domain (game levels, drum patterns, configs, MIDI, visual tokens) can plug in by supplying a corpus and a `RetrievalConfig`. The sparse-attention kernel is used as an associative memory — no autograd, no learned weights, no Python toolchain.

Two pipelines from one kernel:

- `Retriever::generate_fast` — autoregressive next-token retrieval via `KvCache` + `decode_step`, O(log T) per generated token.
- `Diffuser::diffuse` — bidirectional masked discrete diffusion with a MaskGIT cosine schedule. Beats AR on aggregate by 6.9x on the Mario benchmark (see `sparse-mario` baselines doc).

## Layout

- `Cargo.toml` — features `default = ["std"]`, `std = ["ruvllm_sparse_attention/std"]`. Single path dep on `ruvllm_sparse_attention`. Research-tier clippy allow-list.
- `src/lib.rs` — entire crate in one file. Defines `RetrievalConfig`, `Retriever`, `Diffuser`, `SamplingConfig`. Re-exports `SparseAttentionConfig as SparseConfig` from the kernel crate.

## Public API surface

- `RetrievalConfig` — `vocab_size` (≤ 254), `head_dim`, `pos_scale`, `mask_sentinel`, etc.
- `Retriever::new(corpus, cfg, seed)` + `.generate_fast(seed, n, sampling, rng_seed)`.
- `Diffuser` + `.diffuse(...)` with `SamplingConfig::quality()` / similar presets.
- Plug-in checklist for new domains is in the crate-level doc comment.

## Examples

- `examples/drum_patterns.rs` — drum-pattern domain demo (non-text small-vocab plug-in).

## Tests

None in this crate.

## Related crates

- `crates/ruvllm_sparse_attention` — the underlying sub-quadratic sparse attention kernel (`SubquadraticSparseAttention`, `KvCache`, `Tensor3`).
- `sparse-mario` (example workspace) — referenced as the original benchmark.
