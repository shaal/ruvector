# a2a-swarm/src

Single-file orchestrator that drives the A2A swarm demo end-to-end.

## Files

- `main.rs` — tokio binary that:
  1. Resolves the `rvagent` binary path (built via the `rvagent-cli`
     path dep declared in `../Cargo.toml`).
  2. Spawns three nodes from `../configs/` with `kill_on_drop` semantics,
     waiting on each child's stderr until it logs "listening".
  3. Calls `rvagent a2a discover` and `rvagent a2a send-task` against
     the router (`:18003`).
  4. Asserts `Task.metadata.ruvector.routed_via.peer_url` equals one of
     the two leaf URLs — proves the router forwarded over HTTP.
  5. Tears nodes down via SIGTERM with a `SHUTDOWN_TIMEOUT` fallback.

## Key constants

- `NODES` — three `NodeSpec { name, bind, config }` entries.
- `ROUTER_INDEX = 2`, `STARTUP_TIMEOUT = 30s`, `SHUTDOWN_TIMEOUT = 10s`.

## Related

- `../configs/` — the TOML files each node loads
- `../../../crates/rvAgent/rvagent-cli/` — the binary being orchestrated
