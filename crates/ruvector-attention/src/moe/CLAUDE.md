# ruvector-attention/src/moe

Mixture-of-Experts attention: route tokens / queries to specialised expert heads.

## Files

- `mod.rs` — module entry.
- `expert.rs` — `Expert` trait + default expert head.
- `router.rs` — top-k / softmax router.
- `moe_attention.rs` — full MoE-attention forward pass.
