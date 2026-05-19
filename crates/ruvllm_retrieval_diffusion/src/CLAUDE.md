# ruvllm_retrieval_diffusion/src

Single-file crate.

## Files

- `lib.rs` — defines `RetrievalConfig`, `Retriever`, `Diffuser`, `SamplingConfig`. Wires the `ruvllm_sparse_attention::{SubquadraticSparseAttention, KvCache, AttentionBackend, SparseAttentionConfig, Tensor3}` kernel as an associative-memory engine for both autoregressive and masked-diffusion sampling. No autograd, no learned weights.
