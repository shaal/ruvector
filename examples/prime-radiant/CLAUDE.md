# prime-radiant

`prime-radiant-category` crate: advanced mathematical structures for AI interpretability - sheaf cohomology, category theory, topos, Homotopy Type Theory (HoTT), spectral analysis, causal reasoning, and quantum topology. Used as a research substrate for coherent belief modeling and structure-preserving retrieval. Ships a WASM subcrate.

## Files

- `Cargo.toml` - Manifest; features `std` (default), `wasm`, `bench`, `parallel`, `simd`. Many criterion benches gated behind `bench`. Disables auto test/bench discovery.
- `Cargo.lock` - Lockfile.
- `src/` - Library implementation, split into `category/`, `causal/`, `cohomology/`, `hott/`, `quantum/`, `spectral/`, plus `belief.rs`, `topos.rs`, etc.
- `tests/` - Module-level integration tests (most gated/disabled in manifest).
- `benches/` - Criterion benches per module.
- `docs/` - ADRs and DDD docs.
- `wasm/` - WASM bindings subcrate (own workspace).

## How to build/run

```bash
cargo build -p prime-radiant-category --release
cargo test  -p prime-radiant-category --test integration_tests
cargo bench -p prime-radiant-category --features bench --bench category_bench
```

## Tech stack

- Rust 2021. Core deps: `nalgebra`, `ndarray`, `petgraph`, `num-complex`, `dashmap`, `parking_lot`, `serde`, `thiserror`, `uuid`, `rand`, `rand_chacha`, `rand_distr`; optional `rayon`, `wasm-bindgen`, `criterion`.
- Internal: `ruvector-solver` (Neumann, CG).

## Related

- Sibling formal-methods/math demos: `examples/rvf-kernel-optimized`, `examples/rvf` (verified store + math).
- ADRs: `docs/adr/ADR-001..006`.
