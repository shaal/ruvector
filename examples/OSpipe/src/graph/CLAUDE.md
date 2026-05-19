# OSpipe / src / graph

Graph layer over captured content. Extracts entities/relations so the search stage can blend vector similarity with graph-structural context (via `ruvector-graph` / `ruvector-gnn`).

## Important files
- `mod.rs` - module root, public graph API.
- `entity_extractor.rs` - entity extraction from captured frames/text used to build the knowledge graph.

## Related
- Workspace crates wired in via `../../Cargo.toml`: `ruvector-graph`, `ruvector-gnn`.
- Consumed by `../search/enhanced.rs` and `../search/hybrid.rs`.
