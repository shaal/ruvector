# ruqu-exotic

Experimental quantum-classical hybrid algorithms — research-tier crate that exploits the unusual coexistence of a quantum simulator with vector-database primitives. Eight modules cover quantum-memory decay, interference search, reasoning QEC, swarm interference, syndrome diagnosis, reversible memory, and reality-check verification.

## Important files

- `Cargo.toml` — Depends on `ruqu-core` (path), `rand`, `thiserror`, optional `serde`. Features `default = ["std"]`. Lint config relaxes warnings (`research-tier`).
- `src/lib.rs` — Crate root. Documents 8 module concepts and what they replace (e.g. `quantum_decay` replaces TTL eviction; `interference_search` replaces cosine reranking).

## Source modules (`src/`)

- `quantum_decay.rs` — Embeddings decohere instead of being deleted.
- `interference_search.rs` — Concepts interfere during retrieval.
- `quantum_collapse.rs` — Search collapses from superposition.
- `reasoning_qec.rs` — Surface-code correction on reasoning traces.
- `swarm_interference.rs` — Agents interfere instead of voting.
- `syndrome_diagnosis.rs` — QEC syndrome extraction for system diagnosis.
- `reversible_memory.rs` — Time-reversible state for counterfactual debugging.
- `reality_check.rs` — Browser-native quantum verification circuits.

## Tests

- `tests/test_exotic.rs` — Module-by-module sanity tests.
- `tests/test_discovery_cross.rs`, `test_discovery_phase2.rs`, `test_discovery_pipeline.rs` — Discovery-pipeline experiments (these algorithms double as research probes).
- Dev-dep `approx = "0.5"` for float comparisons.

## Related

- Sibling: `ruqu-core` (foundational quantum primitives), `ruqu-wasm` (browser bindings), `ruQu` (classical nervous system).
