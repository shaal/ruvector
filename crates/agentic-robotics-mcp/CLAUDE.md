# agentic-robotics-mcp

Model Context Protocol (MCP) server crate that exposes `agentic-robotics-core` robot capabilities to AI assistants. Implements the
MCP 2025-11 spec with stdio transport by default and an optional SSE (axum) transport.

## Files

- `Cargo.toml` — depends on `agentic-robotics-core`, tokio, anyhow, thiserror, tracing. Optional `sse` feature pulls in `axum` +
  `tokio-stream`. Has `README.md`.
- `src/lib.rs` — public API: `MCP_VERSION` (`"2025-11-15"`), `McpTool`, `McpRequest`, `McpResponse`, and re-exports of the
  `transport` and `server` submodules.
- `src/server.rs` — `McpServer` implementation: JSON-RPC dispatch, tool registry, handler wiring.
- `src/transport.rs` — transport traits + concrete stdio and (feature-gated) SSE transports.

## Features

- `default = []` — stdio only.
- `sse` — enables axum-based Server-Sent Events transport.

## Related

- `../agentic-robotics-core` — message/pubsub/robot core that this MCP exposes.
- `../agentic-robotics-rt` — async runtime sibling.
- `../agentic-robotics-benchmarks` — perf harness for the underlying core.
