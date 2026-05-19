# prime-radiant/benches

Criterion benchmark suite (gated by the `bench` feature in `Cargo.toml`).

## Files

- `category_bench.rs` - Functors, composition chains, topos operations.
- `cohomology_bench.rs` - Coboundary operators, cohomology groups, sheaf neural layers.
- `spectral_bench.rs` - Eigenvalue computation, Cheeger constant, spectral clustering.
- `causal_bench.rs` - Interventions, counterfactuals, causal abstraction.
- `quantum_bench.rs` - Persistent homology, quantum states, density matrices.
- `integrated_bench.rs` - End-to-end coherence, memory profiling, throughput.
- `quantum_solver_bench.rs` - Solver-backed quantum operator benchmarks.

## How to run

```bash
cargo bench -p prime-radiant-category --features bench --bench category_bench
cargo bench -p prime-radiant-category --features bench --bench quantum_bench
```

## Related

- Implementations: `../src/category`, `../src/cohomology`, `../src/spectral`, `../src/causal`, `../src/quantum`.
- Solver: `crates/ruvector-solver`.
