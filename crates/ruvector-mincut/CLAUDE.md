# ruvector-mincut

World's first **subpolynomial-time dynamic minimum cut** library. Maintains minimum cuts under edge insertions/deletions with O(n^{o(1)}) amortized update time (for cuts up to 2^{O((log n)^{3/4})}); also offers (1+ε)-approximate cuts via graph sparsification. Targets self-healing networks, AI optimization, and real-time graph analysis.

## Important files

- `Cargo.toml` — many feature gates and many internal subsystems. Hard deps include `petgraph`, `rayon`, `crossbeam`, `parking_lot`, `dashmap`, `roaring`, `ordered-float`. Optional `ruvector-core`, `ruvector-graph`.
- `src/lib.rs` — top-level docs, `MinCutBuilder`, `DynamicMinCut`, monitoring/event API.
- `src/error.rs` — crate-wide error enum.
- `src/time_compat.rs` — time-source abstraction (host vs. wasm).
- `README.md`, `docs/` — extensive user docs (ALGORITHMS, API, ARCHITECTURE, BENCHMARK_REPORT, PAPER_IMPLEMENTATION, witness docs, sparsification notes).
- `docs/adr/` — ADRs incl. ADR-001 Anytime-Valid Coherence Gate, ADR-002 Dynamic Hierarchical JTree Decomposition (+ addenda for BMSSP / SOTA optimizations), DDC-001, ROADMAP, applications appendix.
- `docs/guide/` — 8-part user guide (getting started → API reference → troubleshooting).
- `docs/security/` — security review (BMSSP).

## Module map (src/)

Algorithm core:
- `algorithm/` — exact (replacement-based) + approximate algorithms.
- `subpolynomial/` — the headline subpolynomial-time scheme.
- `canonical/` — canonical decomposition (dynamic, source-anchored, tree-packing).
- `core/` — shared abstractions.
- `instance/` — bounded / stub instance representations and traits.

Decomposition / graph structure:
- `graph/`, `tree/`, `jtree/`, `linkcut/`, `euler/`, `expander/`, `cluster/`, `connectivity/`, `fragment/`, `fragmentation/`, `compact/`, `sparsify/`.

Local / advanced:
- `localkcut/` — deterministic local k-cut discovery (4-color coding).
- `certificate/`, `witness/` — proof / audit chain.
- `snn/` — spiking neural-network cognitive engine layer.
- `optimization/` — caching, parallel, SIMD distance, dspar, wasm batch.
- `integration/`, `wrapper/`, `wasm/` — outward integration points.
- `parallel/`, `pool/`, `monitoring/` — runtime perf and event-driven monitoring.

## Tests & benches

- `tests/` — `bounded_integration`, `canonical_bench`, `certificate_tests`, `coverage_tests`, `integration_tests`, `jtree_tests`, `localkcut_*`, `paper_algorithm_tests`, `wrapper_tests`.
- `benches/` — `bounded_bench`, `canonical_bench`, `jtree_bench`, `mincut_bench`, `optimization_bench`, `paper_algorithms_bench`, `snn_bench`, `sota_bench`.
- `examples/` — `localkcut_demo.rs`, `sparsify_demo.rs`, `subpoly_bench.rs`.

## Public API surface

`MinCutBuilder`, `DynamicMinCut`, monitoring events, plus the canonical / local-k-cut / sparsifier / witness submodule types.

## Related

- `crates/ruvector-mincut-wasm` — wasm bindings.
- `crates/ruvector-mincut-gated-transformer`, `crates/ruvector-attn-mincut` — transformer integrations that rely on min-cut.
- `crates/cognitum-gate-tilezero`, `crates/mcp-gate` — coherence-gate stack referenced by ADR-001.
