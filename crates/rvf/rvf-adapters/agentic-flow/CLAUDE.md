# rvf-adapter-agentic-flow

RVF adapter for agentic-flow swarm coordination (ADR-029). Persists inter-agent memory sharing, swarm coordination state, and learning patterns.

Segment mapping:
- **VEC_SEG + META_SEG**: shared memory entries (embedding + key/value metadata)
- **META_SEG**: swarm coordination state (agent states, topology)
- **SKETCH_SEG**: agent learning patterns with effectiveness scores
- **WITNESS_SEG**: distributed consensus votes with signatures

## Layout

- `Cargo.toml` — name `rvf-adapter-agentic-flow`. Deps: `rvf-runtime`, `rvf-types`, `rvf-crypto` (all `std`). Dev: `tempfile`.
- `src/lib.rs` — public `AgenticFlowConfig`, `RvfSwarmStore` (+ inner module exports).
- `src/config.rs` — `AgenticFlowConfig::new(data_dir, agent_id)`.
- `src/swarm_store.rs` — `RvfSwarmStore` — main store API.
- `src/coordination.rs` — agent-state/topology records (META_SEG).
- `src/learning.rs` — learning patterns + effectiveness scores (SKETCH_SEG).

## Related

- `../../rvf-runtime`, `../../rvf-types`, `../../rvf-crypto`
- Sibling adapters under `../`
