# exo-backend-classical/src

Library source for the classical compute backend.

## Files

- `lib.rs` — public surface, re-exports, `ClassicalBackend` type.
- `vector.rs` — wraps `ruvector-core` vector primitives.
- `graph.rs` — wraps `ruvector-graph` graph primitives.
- `dither_quantizer.rs` — exposes `ruvector-dither` quantization.
- `thermo_layer.rs` — Landauer/`thermorust` accounting glue.
- `domain_bridge.rs` — `ruvector-domain-expansion` adapter.
- `transfer_orchestrator.rs` — orchestrates cross-domain transfers.

## Related

- `../tests/` — unit + perf comparison tests for each module above.
- `../../exo-core/src/traits.rs` — implemented contracts.
