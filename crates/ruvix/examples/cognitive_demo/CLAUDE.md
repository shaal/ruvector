# ruvix-demo (cognitive_demo)

Comprehensive RVF package demonstrating ALL RuVix kernel features per ADR-087. Implements a complete cognitive pipeline:

- **5 Components**: SensorAdapter, FeatureExtractor, ReasoningEngine, Attestor, Coordinator.
- **3 Region Types**: Immutable, AppendOnly, Slab.
- **3 Proof Tiers**: Reflex, Standard, Deep.
- **All 12 Syscalls**: demonstrated with proper capability and proof gating.

## Files

- `Cargo.toml` — depends on `ruvix-types`, `ruvix-nucleus`, `ruvix-region`, `ruvix-cap`, `ruvix-queue`, `ruvix-proof`,
  `ruvix-boot`. `crate-type = ["rlib"]`.
- `README.md` — public docs.
- `src/` — see `src/CLAUDE.md`.
- `tests/` — `feature_coverage.rs`, `full_pipeline.rs`.
- `benches/pipeline_bench.rs` — end-to-end pipeline latency.
- `examples/cognitive_demo.rs` — runnable example main.
