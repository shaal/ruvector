# ruvector-dag/src/attention

Graph-topology-aware attention mechanisms for DAG-based query optimisation. Unlike pure neural attention, these use structural DAG properties (topology, paths, cuts, branches) to compute per-node importance.

## Base mechanisms (Team 2 / Agent #2)

- `mod.rs` — module wiring and re-exports.
- `traits.rs` — `DagAttention` trait, `AttentionConfig`, `AttentionScores`, `AttentionError`.
- `trait_def.rs` — extended trait definitions for advanced mechanisms.
- `topological.rs` — `TopologicalAttention`, `TopologicalConfig`.
- `causal_cone.rs` — `CausalConeAttention`, `CausalConeConfig` (forward / backward cones).
- `critical_path.rs` — `CriticalPathAttention`, `CriticalPathConfig` (longest-path weighting).
- `mincut_gated.rs` — `MinCutGatedAttention`, `MinCutConfig`, `FlowCapacity` (gates via mincut signal).

## Advanced mechanisms (Team 2 / Agent #3)

- `parallel_branch.rs` — parallel-branch attention across independent sub-DAGs.
- `temporal_btsp.rs` — temporal "binding-time spike propagation" attention.
- `hierarchical_lorentz.rs` — hyperbolic / Lorentz-space hierarchical attention.
- `cache.rs` — attention-score cache.
- `selector.rs` — runtime selection between attention strategies.

## Other

- `IMPLEMENTATION_NOTES.md` — design notes per mechanism.

Public re-exports flow up through `mod.rs` -> `../lib.rs`. See `../CLAUDE.md`.
