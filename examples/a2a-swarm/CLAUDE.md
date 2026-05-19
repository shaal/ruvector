# a2a-swarm

End-to-end demo of rvAgent's Agent-to-Agent (A2A) protocol: spawns three
`rvagent a2a serve` child processes on local ports (`:18001`, `:18002`,
`:18003`), one of which is a router that discovers the other two as peers
and forwards a task using a `CheapestUnderLatency` selector. Working
orchestrator binary — the assertion proves the router actually forwarded
over HTTP rather than handling locally.

## Important files

- `Cargo.toml` — declares the `a2a-swarm` bin and pulls in `rvagent-cli`
  from `../../crates/rvAgent/rvagent-cli` so building the demo also
  builds the `rvagent` binary it shells out to.
- `src/main.rs` — orchestrator: spawns nodes via `tokio::process`, waits
  for them to bind, calls `a2a discover` / `a2a send-task` subcommands,
  asserts the routed-via metadata, and tears everything down via
  `kill_on_drop`.
- `configs/` — three TOML node configs (cheap/fast/router) with distinct
  `[policy]`, `[budget]`, `[recursion]`, and (for the router)
  `[routing]` + `[[routing.peers]]` blocks.

## Run

```bash
cargo run -p a2a-swarm
```

Logs through `tracing` (`RUST_LOG` honored via `EnvFilter`).

## Tech stack

- Rust async (tokio multi-thread runtime, `tokio::process` for children)
- `reqwest` (rustls) — not actively used in the wrapper (CLI does HTTP)
- `tracing` + `tracing-subscriber`
- Sibling crate: `../../crates/rvAgent/rvagent-cli` (provides
  `a2a serve`, `a2a discover`, `a2a send-task` subcommands)

## Related

- `../../crates/rvAgent/` — full A2A protocol implementation
- Other examples generally focus on vector/graph algorithms; this is the
  only A2A orchestration demo in this chunk.
