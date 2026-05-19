# agentic-robotics-mcp/src

MCP 2025-11 server source for the agentic-robotics ecosystem.

## Files

- `lib.rs` — crate entry; defines `MCP_VERSION`, JSON-RPC envelope types (`McpRequest`, `McpResponse`), and the `McpTool`
  description struct. Re-exports `server` and `transport`.
- `server.rs` — `McpServer` that registers tools, dispatches JSON-RPC methods, and bridges to the underlying robot core via shared
  `Arc<RwLock<...>>` state.
- `transport.rs` — pluggable transport layer: stdio always available; SSE/axum behind the `sse` feature flag.
