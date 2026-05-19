# rvf-adapter-agentic-flow/src

Source.

## Files

- `lib.rs` — public re-exports + ADR-029 segment-mapping docs.
- `config.rs` — `AgenticFlowConfig { data_dir, agent_id, ... }`.
- `swarm_store.rs` — `RvfSwarmStore` main API (create/open/share memory/query).
- `coordination.rs` — agent-state and topology records (META_SEG).
- `learning.rs` — agent learning patterns with effectiveness scores (SKETCH_SEG); witness consensus via `rvf-crypto`.
