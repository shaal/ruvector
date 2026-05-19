# rvagent-mcp/src

Source for the MCP integration.

- `lib.rs` — module roots and re-exports (`McpClient`, `ToolFilter`,
  `ToolGroup`, JSON-RPC types, `McpMethod`, etc.).
- `main.rs` — standalone MCP server binary entry.
- `protocol.rs` — JSON-RPC 2.0 / MCP types.
- `registry.rs` — thread-safe tool registry + handler dispatch.
- `resources.rs` — `static`, `file`, and `template` resource providers + registry.
- `transport.rs` — `stdio` and in-memory transports.
- `server.rs` — MCP server routing to tools/resources.
- `client.rs` — `McpClient` for outbound MCP calls.
- `middleware.rs` — wire MCP into the rvAgent middleware pipeline.
- `groups.rs` — `ToolGroup` and `ToolFilter`.
- `topology.rs` — multi-agent routing topology strategies.
- `skills_bridge.rs` — Claude Code / Codex skills format bridge.
