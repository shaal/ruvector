# rvf-adapters

Sibling directory grouping the RVF adapter crates that bridge external systems onto the RuVector Format segment model (ADR-029). Each subdir is its own workspace member listed in `../Cargo.toml`.

## Subdirs

- `agentdb/` — `rvf-adapter-agentdb`: AgentDB vector/index/pattern storage ↔ VEC_SEG/INDEX_SEG/META_SEG.
- `agentic-flow/` — `rvf-adapter-agentic-flow`: swarm coordination, shared memory, learning patterns, consensus witnesses.
- `claude-flow/` — `rvf-adapter-claude-flow`: claude-flow memory subsystem with WITNESS_SEG audit chain.
- `ospipe/` — `rvf-adapter-ospipe`: observation-state pipeline (screen/audio/UI embeddings) ↔ VEC_SEG/META_SEG/JOURNAL_SEG.
- `rvlite/` — `rvf-adapter-rvlite`: minimal embedded vector-store API over RVF Core Profile (no metadata/filters).
- `sona/` — `rvf-adapter-sona`: SONA trajectories, experience-replay buffers, neural patterns in a single RVF file.

## Related

- `../rvf-runtime` — common backend (`RvfStore`)
- `../rvf-types`, `../rvf-crypto` — shared segment / crypto primitives
