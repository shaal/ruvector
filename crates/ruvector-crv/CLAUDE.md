# ruvector-crv

CRV (Coordinate Remote Viewing) protocol integration for ruvector. Maps the 6-stage CRV signal-line methodology to ruvector subsystems:

| Stage | Data | Component |
|---|---|---|
| I — Ideograms | Gestalt primitives | Poincare hyperbolic embeddings |
| II — Sensory | Textures, colors, temps | Multi-head attention vectors |
| III — Dimensional | Spatial sketches | GNN graph topology |
| IV — Emotional | AOL, intangibles | SNN temporal encoding |
| V — Interrogation | Signal-line probing | Differentiable search |
| VI — 3D Model | Composite | MinCut partitioning |

## Layout

- `Cargo.toml` — deps: `ruvector-attention`, `ruvector-gnn` (default-features=false), `ruvector-mincut` (with `exact`), `serde`, `serde_json`, `thiserror`.
- `src/lib.rs` — module declarations + `CrvSessionManager` quick-start example.
- `src/session.rs` — `CrvSessionManager`: per-session state across all 6 stages.
- `src/types.rs` — shared value objects, including `GestaltType`.
- `src/error.rs` — `CrvError`.
- `src/stage_i.rs` — `StageIData` (ideograms): Poincare ball embedding of the gestalt taxonomy.
- `src/stage_ii.rs` — sensory descriptor encoding via multi-head attention.
- `src/stage_iii.rs` — spatial sketch → GNN topology.
- `src/stage_iv.rs` — emotional / AOL signal via SNN temporal encoding (high-freq bursts = AOL, sustained low-freq = clean signal).
- `src/stage_v.rs` — signal-line interrogation via differentiable search.
- `src/stage_vi.rs` — composite 3D model via MinCut partitioning.

Default config: 384-dim embeddings.

## Tests / examples

- Dev-deps: `approx`. No `tests/` or `examples/` folder.

## Related crates

- `crates/ruvector-attention` (Stage II)
- `crates/ruvector-gnn` (Stage III)
- `crates/ruvector-mincut` (Stage VI)
- See ADRs covering CRV integration in workspace docs.
