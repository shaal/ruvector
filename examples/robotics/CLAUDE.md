# robotics

`ruvector-robotics-examples` crate: a numbered tutorial series (10 binaries) demonstrating the `ruvector-robotics` crate - perception, planning, scene graphs, behavior trees, swarm coordination, skill learning, world models, and MCP tool integration.

## Files

- `Cargo.toml` - Manifest with 10 `[[bin]]` entries (`01_basic_perception` ... `10_full_pipeline`). Depends on `ruvector-robotics`, `serde_json`, `rand`.
- `src/main.rs` - Small umbrella entry / shared helpers.
- `src/bin/01..10_*.rs` - The 10 numbered example programs.

## How to run

```bash
cargo run -p ruvector-robotics-examples --bin 01_basic_perception
cargo run -p ruvector-robotics-examples --bin 05_cognitive_robot
cargo run -p ruvector-robotics-examples --bin 10_full_pipeline
```

## Tech stack

- Rust 2021. Internal crate `ruvector-robotics`; `serde_json`, `rand`.

## Related

- Underlying crate: `crates/ruvector-robotics`.
- Behavior tree / agent demos elsewhere: `examples/rvf/examples/agent_handoff.rs`, `examples/a2a-swarm`.
