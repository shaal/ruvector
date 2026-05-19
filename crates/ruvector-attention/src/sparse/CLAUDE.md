# ruvector-attention/src/sparse

Sparse attention patterns (local windows, global tokens, linear, flash).

## Files

- `mod.rs` — module entry.
- `mask.rs` — sparse attention mask types + helpers.
- `local_global.rs` — local window + global token pattern (Longformer-style).
- `linear.rs` — linear-complexity attention (e.g. kernelised).
- `flash.rs` — sparse variant of flash-attention.
