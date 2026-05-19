# ruvector-robotics/src

Crate root.

## Files

- `lib.rs` — top-level docs, module declarations: `bridge`, `cognitive`, `mcp`, `perception`, `planning`. Feature-gated re-exports for `domain_expansion` and `rvf`.
- `bridge/` — foundational types: point clouds, poses, spatial index, perception pipeline glue.
- `perception/` — scene graph, obstacle detection, sensor fusion, clustering.
- `cognitive/` — behaviour trees, memory tiers, decision engine, skills, swarm, world model.
- `mcp/` — Model Context Protocol tool registry + executor.
- `planning.rs` — task/path planning utilities used by cognitive layers.
- `domain_expansion.rs` — `#[cfg(feature = "domain-expansion")]` integration with `ruvector-domain-expansion`.
- `rvf.rs` — `#[cfg(feature = "rvf")]` persistence via RVF segments.
