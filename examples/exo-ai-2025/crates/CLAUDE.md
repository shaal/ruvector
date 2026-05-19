# exo-ai-2025/crates

Nine workspace member crates that make up the EXO-AI substrate. All
share `[workspace.package]` versioning (`0.1.x`) and are wired up by
`../Cargo.toml`.

## Crates

- `exo-core/` — core traits/types (`Substrate`, `Witness`,
  `Consciousness`, `Thermodynamics`, `CoherenceRouter`); IIT + Landauer
  primitives. Used by every other crate here.
- `exo-hypergraph/` — hyperedge data structures, sheaf cohomology,
  sparse TDA / topology helpers.
- `exo-manifold/` — continuous embedding via SIREN nets, manifold
  deformation, retrieval, forgetting.
- `exo-temporal/` — short/long-term memory, causal links, quantum
  decay, anticipation/consolidation.
- `exo-federation/` — distributed cognitive mesh: CRDTs, Byzantine
  consensus, post-quantum handshake, onion routing.
- `exo-backend-classical/` — SIMD-accelerated classical compute backend
  glueing the above to `ruvector-core`/`ruvector-graph` and
  `thermorust`.
- `exo-exotic/` — research experiments: strange loops, dreams, free
  energy, morphogenesis, collective consciousness, black holes,
  thermodynamics, emergence detection.
- `exo-node/` — Node.js bindings via NAPI-RS (`cdylib`).
- `exo-wasm/` — browser/edge WASM bindings (`cdylib`+`rlib`).

## Build

```bash
cargo build -p exo-core
cargo build --workspace        # from ../
```

## Related

- `../tests/` — workspace-level integration tests across crates
- `../docs/`, `../architecture/`, `../specs/`
