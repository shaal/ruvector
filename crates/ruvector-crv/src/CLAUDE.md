# ruvector-crv/src

Flat-file layout: one file per CRV stage plus shared session / types / error.

## Files

- `lib.rs` — crate doc + module wiring; quick-start example.
- `session.rs` — `CrvSessionManager` (create / lookup sessions; per-stage `add_*` methods).
- `types.rs` — shared types, including `GestaltType` enum.
- `error.rs` — `CrvError`.
- `stage_i.rs` — Ideograms (`StageIData`, Poincare gestalt embedding).
- `stage_ii.rs` — Sensory (multi-head attention vectors).
- `stage_iii.rs` — Dimensional (GNN graph topology from spatial sketches).
- `stage_iv.rs` — Emotional / AOL (SNN temporal encoding).
- `stage_v.rs` — Interrogation (differentiable search probing).
- `stage_vi.rs` — 3D Model (MinCut composite).
