# ruvector-robotics

Unified cognitive robotics platform built on the ruvector vector DB and graph/neural infrastructure. Bundles bridge types (point clouds, poses, spatial index), a perception stack (scene graph, obstacle detection), a cognitive architecture (behavior trees, memory, skills, swarm), and MCP tool registrations.

## Layout

- `Cargo.toml` — three feature flags: `default = []`, `domain-expansion` (pulls in `../ruvector-domain-expansion` + `rand`), `rvf` (pulls in `../rvf/rvf-runtime`, `../rvf/rvf-types`, `tempfile`). Dev-deps: `criterion`. Lints relaxed (research-tier).
- `src/lib.rs` — top-level docs + module re-exports.
- `src/bridge/` — robotics primitives, converters, gaussian helpers, spatial indexing, perception pipeline. See `src/bridge/CLAUDE.md`.
- `src/perception/` — scene graph, obstacle detector, sensor fusion, clustering. See `src/perception/CLAUDE.md`.
- `src/cognitive/` — behavior tree, decision engine, memory, skill learning, swarm, world model. See `src/cognitive/CLAUDE.md`.
- `src/mcp/` — MCP tool registry + executor. See `src/mcp/CLAUDE.md`.
- `src/planning.rs` — high-level path/task planning utilities.
- `src/domain_expansion.rs` — gated on `domain-expansion` feature; bridge to `ruvector-domain-expansion`.
- `src/rvf.rs` — gated on `rvf` feature; persistence/replay via RVF.
- `benches/robotics_benchmarks.rs` — Criterion bench for cloud conversion, kNN, obstacle pipeline, trajectory prediction, attention, BT tick, scene graph build.
- `examples/` — `behavior_tree`, `cognitive_loop`, `obstacle_detection`, `spatial_indexing`, `swarm_coordination`.
- `tests/` — `integration.rs`, `robotics_integration.rs`.

## Public API

`bridge::{Point3D, PointCloud, SpatialIndex, RobotState, Pose, Trajectory}`, the perception/cognitive/mcp module trees, optional `domain_expansion`/`rvf` shims.

## Related

- `../ruvector-domain-expansion` — feature-gated dependency
- `../rvf/rvf-runtime`, `../rvf/rvf-types` — feature-gated dependency
- `../agentic-robotics-core`, `../agentic-robotics-node`, `../agentic-robotics-embedded` — companion robotics surface focused on pub/sub
