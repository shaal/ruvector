# ruvector-mincut/src/canonical

Canonical decomposition of min-cuts: factors a cut into invariant components that can be maintained more cheaply under dynamic updates.

## Files

- `mod.rs` — façade and re-exports.
- `tests.rs` — module-level unit tests for the canonical decomposition.

## Subdirectories

- `dynamic/` — dynamic maintenance of the canonical decomposition.
- `source_anchored/` — source-anchored (single-source) canonical decomposition.
- `tree_packing/` — tree-packing-based decomposition (Gabow-style).
