# mcp-gate/src

MCP stdio server source.

## Files

- `lib.rs` — crate doc + module declarations + public re-exports (`McpGateServer`, `McpGateConfig`).
- `main.rs` — binary entrypoint; configures tracing, builds `McpGateConfig` from env, runs `server.run_stdio()`.
- `server.rs` — `McpGateServer`, `ServerInfo`; JSON-RPC 2.0 loop over stdin/stdout using `tokio::io::{AsyncBufReadExt, AsyncWriteExt}`. Routes MCP requests to `McpGateTools`.
- `tools.rs` — `McpGateTools` holds an `Arc<RwLock<TileZero>>` and implements:
  - `permit_action` — request permission, returns PermitToken / escalation / denial.
  - `get_receipt` — fetch a signed receipt by sequence number.
  - `replay_decision` — deterministically replay a past decision (optionally verifying the hash chain).
- `types.rs` — JSON-RPC request/response/error types and tool argument/result schemas.
