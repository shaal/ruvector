# ruvector-nervous-system/src/routing

Cognitive routing — direct information between modules based on state.

## Files

- `mod.rs` — façade.
- `circadian.rs` — circadian-rhythm routing (time-of-day modulation).
- `coherence.rs` — coherence-driven routing (uses min-cut / coherence-gate signals).
- `predictive.rs` — predictive routing using forward models.
- `workspace.rs` — Global Workspace Theory broadcast (used by `mcp-brain-server` cognitive loop).
