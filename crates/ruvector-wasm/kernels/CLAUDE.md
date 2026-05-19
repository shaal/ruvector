# ruvector-wasm/kernels

First-party ML kernel sources targeted for the kernel pack system. These are
standalone Rust files that get built into WASM kernel packs and consumed by
the sandbox runtime in `src/kernel/`.

## Files
- `rmsnorm.rs` - RMSNorm kernel (root-mean-square layer normalization).
- `rope.rs` - RoPE kernel (rotary position embeddings).
- `swiglu.rs` - SwiGLU kernel (Swish-gated linear unit activation).
