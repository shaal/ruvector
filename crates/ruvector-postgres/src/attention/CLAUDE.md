# ruvector-postgres/src/attention

Implements 39 attention mechanisms as PostgreSQL SQL functions/operators:
- Core: Scaled dot-product, Multi-head, Flash Attention v2
- Graph: GAT, GATv2, Sparse patterns
- Specialized: MoE, Cross-attention, Sliding window
- Hyperbolic: Poincaré, Lorentzian attention

SIMD-accelerated with efficient memory usage.

## Files

- `mod.rs` — Module entry: type defs (Serialize/Deserialize), shared helpers; declares `flash` submodule.
- `flash.rs` — Flash Attention v2 implementation.
- `multi_head.rs` — Multi-head attention.
- `scaled_dot.rs` — Scaled dot-product attention.
- `operators.rs` — pgrx-exposed SQL operators wrapping the above.
