# prime-radiant/src/simd

SIMD inner loops used by the coherence engine's hot paths.

## Files

- `mod.rs` — module entry; arch dispatch.
- `vectors.rs` — vector-vector ops (add, dot, axpy).
- `matrix.rs` — small dense matrix * vector for restriction maps.
- `energy.rs` — fused residual + squared-norm + weight reduction.

Pairs with `gpu/` for the alternative GPU path.
