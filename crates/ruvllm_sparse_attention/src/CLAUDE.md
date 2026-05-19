# ruvllm_sparse_attention/src

- `lib.rs` — crate root. `#![cfg_attr(not(feature = "std"), no_std)]` + `extern crate alloc`. On no_std, defines `no_std_math::F32Ext` so call sites can keep using `.exp() / .sqrt() / .tanh() / .powi()` via `libm`. Re-exports the public API from the four submodules.
- `attention.rs` — core kernels: `dense_attention`, `AttentionBackend` trait, `AttentionError`, `SparseAttentionConfig`, `SubquadraticSparseAttention`, `IncrementalLandmarks`, `KvCache` (and `KvCacheF16` under `fp16`).
- `fastgrnn_gate.rs` — `FastGrnnGate` salience gate that turns the kernel near-linear; `DEFAULT_HIDDEN_DIM` re-exported as `FASTGRNN_DEFAULT_HIDDEN_DIM`.
- `model.rs` — `RuvLlmSparseBlock`, `RuvLlmSparseBlockConfig` — a ready-to-use transformer block plugging attention + gate together.
- `tensor.rs` — `Tensor3` minimal allocated tensor type used by the kernels.

See `../CLAUDE.md`.
