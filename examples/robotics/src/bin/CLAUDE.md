# robotics/src/bin

Numbered, progressively-richer tutorial binaries for `ruvector-robotics`.

## Files

- `01_basic_perception.rs` - Sensor input -> perception graph basics.
- `02_obstacle_avoidance.rs` - Reactive obstacle avoidance.
- `03_scene_graph.rs` - Scene graph construction.
- `04_behavior_tree.rs` - Behavior tree planner.
- `05_cognitive_robot.rs` - Cognitive/agentic loop on top of perception+planning.
- `06_swarm_coordination.rs` - Multi-robot coordination.
- `07_skill_learning.rs` - Skill learning / experience replay.
- `08_world_model.rs` - World-model building.
- `09_mcp_tools.rs` - MCP tool integration for an agentic robot.
- `10_full_pipeline.rs` - End-to-end pipeline tying everything together.

## How to run

```bash
cargo run -p ruvector-robotics-examples --bin 03_scene_graph
```

## Related

- Manifest: `../../Cargo.toml`.
- Crate: `crates/ruvector-robotics`.
