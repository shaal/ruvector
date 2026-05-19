# exo-backend-classical

Classical (non-quantum / non-neuromorphic) compute backend for the
EXO-AI substrate. Wires SIMD-accelerated `ruvector-core`/`ruvector-graph`
into the EXO trait set so workloads can run on commodity CPUs while
optional backends (quantum stub, neuromorphic) live in `exo-core`.

## Files

- `Cargo.toml` — depends on `exo-core`, `exo-manifold`, `exo-temporal`,
  `exo-federation`, `exo-exotic`, `ruvector-core` (SIMD feature),
  `ruvector-graph`, plus `thermorust` and `ruvector-dither`.
- `src/lib.rs` — re-exports + the public `ClassicalBackend` impl.
- `src/vector.rs` — vector ops bridged to `ruvector-core`.
- `src/graph.rs` — graph ops bridged to `ruvector-graph`.
- `src/dither_quantizer.rs` — quantization via `ruvector-dither`.
- `src/thermo_layer.rs` — thermodynamic accounting layer
  (`thermorust`).
- `src/domain_bridge.rs` — `ruvector-domain-expansion` integration.
- `src/transfer_orchestrator.rs` — coordinates state transfer across
  domains.
- `tests/` — `classical_backend_test.rs`, `learning_benchmarks.rs`,
  `performance_comparison.rs`, `transfer_pipeline_test.rs`.

## Build / Test

```bash
cargo build -p exo-backend-classical
cargo test  -p exo-backend-classical
```

## Related

- `../exo-core/` — trait set being implemented
- `../../../../crates/thermorust/`, `../../../../crates/ruvector-dither/`
