# exo-manifold

Continuous embedding space + smooth manifold deformation using SIREN
(sinusoidal-activation implicit neural representation) networks. Lets
the substrate retrieve and reshape memory regions as differentiable
manifolds rather than discrete vectors.

## Files

- `Cargo.toml` — depends on `exo-core`, `ruvector-domain-expansion`,
  ndarray, serde, thiserror, parking_lot; dev-dep `approx`.
- `src/lib.rs` — re-exports.
- `src/network.rs` — SIREN network impl.
- `src/deformation.rs` — manifold deformation operators.
- `src/retrieval.rs` — nearest-point / continuous retrieval.
- `src/forgetting.rs` — manifold-level forgetting / pruning.
- `src/simd_ops.rs` — SIMD-accelerated numeric kernels.
- `src/transfer_store.rs` — cross-domain transfer integration.
- `tests/manifold_engine_test.rs` — engine integration tests.

## Build / Test

```bash
cargo build -p exo-manifold
cargo test  -p exo-manifold
```

## Related

- `../../docs/MANIFOLD_IMPLEMENTATION.md`
- `../../benches/manifold_bench.rs`
