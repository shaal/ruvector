# prime-radiant

Universal coherence engine using sheaf-Laplacian mathematics for AI safety, hallucination detection, and structural consistency verification in LLMs and distributed systems. The crate provides a single underlying "coherence object" (residual energy on edges of a substrate graph) reinterpreted for many domains (LLM facts, finance, robotics, security, science).

Pull-request worthy claim: it is NOT prediction — it is a continuously updated coherence field showing where action is safe and where it must stop.

## Architecture (layered)

```
APPLICATION  -> LLM Guards | Fraud | Compliance | Robotics
COHERENCE GATE -> Lane 0 (reflex) / 1 (retrieval) / 2 (heavy) / 3 (human)
COHERENCE COMPUTATION -> Residual + Energy + Spectral + Fingerprints
GOVERNANCE   -> Policy bundles, witness, lineage, tuning
```

## Layout

- `Cargo.toml` — large manifest (17 KB) wiring almost the entire ruvector ecosystem behind optional features (cognitum-gate-kernel, sona, gnn, mincut, hyperbolic-hnsw, nervous-system, attention, raft, core). Per ADR-014 full-ecosystem integration.
- `src/lib.rs` — crate root with extensive module docs and the architecture diagram. Re-exports the engine, gate, governance, witness, and policy types.
- `src/error.rs`, `src/events.rs`, `src/types.rs` — shared error type, event bus, and value-object types.
- `src/coherence/` — core sheaf-Laplacian residual + energy engine.
- `src/cohomology/` — H^0/H^1 sheaf cohomology, coboundary, obstruction detection.
- `src/substrate/` — graph substrate: nodes, edges, restriction maps, repository.
- `src/tiles/` — 256-tile coherence fabric integration (cognitum-gate-kernel).
- `src/attention/` — adapters into `ruvector-attention` (topology-gated, MoE, PDE diffusion).
- `src/mincut/` — adapter into `ruvector-mincut` for cognitive partitioning.
- `src/hyperbolic/` — hierarchy-aware Poincare embeddings & depth.
- `src/learned_rho/` — learned restriction maps (GNN-trained).
- `src/neural_gate/` — neural gating decisions & encoding.
- `src/sona_tuning/` — self-optimizing thresholds via sona EWC++.
- `src/distributed/` — distributed coherence state (Raft adapter).
- `src/execution/` — 4-lane executor / action ladder / gate dispatch.
- `src/governance/` — policy bundles, witness records, lineage repository.
- `src/ruvllm_integration/` — bridge into ruvllm (memory layer, validators, pattern bridge).
- `src/signal/` — input ingestion, normalization, validation.
- `src/security/` — input limits & validation.
- `src/storage/` — pluggable backends: memory, file, postgres.
- `src/simd/` — SIMD kernels for energy/matrix/vector ops.
- `src/gpu/` — wgpu GPU acceleration (residuals, energy, spectral, shaders).

## Tests

- `tests/integration/` — coherence, gate, governance, graph integration tests.
- `tests/property/` — proptest-style invariants.
- Top-level `tests/`: `chaos_tests.rs`, `gpu_coherence_tests.rs`, `replay_determinism.rs`, `ruvllm_integration_tests.rs`, `storage_tests.rs`.

## Benches

`benches/` — attention, coherence, energy, gate, gpu, hyperbolic, incremental, mincut, residual, simd, sona, tile benchmarks (criterion).

## Examples

`examples/basic_coherence.rs`, `compute_ladder.rs`, `governance_audit.rs`, `llm_validation.rs`, `memory_tracking.rs`.

## Docs

`docs/GOAP_ADVANCED_MATH_FRAMEWORKS.md` — advanced math framework notes.

## Related crates

- `cognitum-gate-kernel`, `ruvector-sona`, `ruvector-gnn`, `ruvector-mincut`, `ruvector-hyperbolic-hnsw`, `ruvector-nervous-system`, `ruvector-attention`, `ruvector-raft`, `ruvector-core` (all optional path deps in workspace).
