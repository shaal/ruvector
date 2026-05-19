# edge / src / bin

Three CLI binaries for the edge crate. All are registered in `../../Cargo.toml`.

## Important files
- `agent.rs` - `edge-agent` binary. Runs a single agent that joins a swarm and exchanges messages over the configured transport.
- `coordinator.rs` - `edge-coordinator` binary. Boots a coordinator node that manages routing/topology for the swarm.
- `demo.rs` - `edge-demo` binary. End-to-end demo that spins up agents + coordinator together for a self-contained walkthrough.

## Run
- `cargo run -p ruvector-edge --bin edge-agent`.
- `cargo run -p ruvector-edge --bin edge-coordinator`.
- `cargo run -p ruvector-edge --bin edge-demo`.

## Related
- Library glue: `../intelligence.rs`, `../protocol.rs`, `../transport.rs`. Cargo examples covering similar ground: `../../examples/local_swarm.rs`, `../../examples/distributed_learning.rs`.
