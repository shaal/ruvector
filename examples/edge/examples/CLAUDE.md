# edge / examples

Cargo `[[example]]` programs for the edge crate.

## Important files
- `local_swarm.rs` - boots a small in-process swarm using the shared-memory transport to demonstrate `ruv-swarm-transport` integration.
- `distributed_learning.rs` - distributed-learning demo across multiple agents over the WebSocket transport.

## Run
- `cargo run -p ruvector-edge --example local_swarm`.
- `cargo run -p ruvector-edge --example distributed_learning`.

## Related
- CLI binaries doing similar things: `../src/bin/agent.rs`, `coordinator.rs`, `demo.rs`. Library backing them: `../src/intelligence.rs`, `../src/protocol.rs`, `../src/transport.rs`.
