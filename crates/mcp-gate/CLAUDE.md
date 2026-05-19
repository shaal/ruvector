# mcp-gate

MCP (Model Context Protocol) server exposing the Anytime-Valid Coherence Gate to AI agents. Runs JSON-RPC 2.0 over stdio; lets agents request permissions and obtain cryptographic decision receipts from `cognitum-gate-tilezero`.

## Important files

- `Cargo.toml` — library + `mcp-gate` binary. Deps: `cognitum-gate-tilezero`, `tokio`, `async-trait`, `serde`, `tracing`, `thiserror`.
- `src/lib.rs` — public crate root, documents the three MCP tools (`permit_action`, `get_receipt`, `replay_decision`). Re-exports `McpGateServer`, `McpGateConfig`.
- `src/main.rs` — `mcp-gate` binary; reads config from env vars (`MCP_GATE_TAU_DENY`, etc.) and runs `server.run_stdio()`.
- `src/server.rs` — `McpGateServer` and `ServerInfo`; JSON-RPC stdio loop (tokio AsyncBufRead/AsyncWrite).
- `src/tools.rs` — `McpGateTools`: implements the three tool calls and wraps a `TileZero` instance.
- `src/types.rs` — MCP request/response/error types (serde).

## Public API surface

- `McpGateServer`, `McpGateServer::new()`, `McpGateServer::with_thresholds(GateThresholds)`, `run_stdio()`.
- `McpGateConfig` (env-driven thresholds).
- Re-exports from `cognitum-gate-tilezero`: `TileZero`, `GateThresholds`.

## Related

- `crates/cognitum-gate-tilezero` — the underlying coherence gate (TileZero variant).
- ADRs in `crates/ruvector-mincut/docs/adr/ADR-001-anytime-valid-coherence-gate.md` describe the gate semantics.
