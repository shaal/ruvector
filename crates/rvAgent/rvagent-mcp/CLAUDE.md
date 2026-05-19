# rvagent-mcp

Model Context Protocol (MCP) integration for rvAgent. Provides JSON-RPC 2.0
protocol types, a thread-safe tool registry, resource providers (static / file
/ template), stdio + in-memory transports, server + client, middleware for the
rvagent pipeline, topology strategies for multi-agent routing, and a skills
bridge (Claude Code / Codex formats).

## Layout

- `Cargo.toml` — lib + bin (`src/main.rs`).
- `src/lib.rs` — module roots and re-exports.
- `src/main.rs` — standalone MCP server binary.
- `src/protocol.rs` — JSON-RPC 2.0 types (`JsonRpcRequest`/`Response`/`Error`),
  `McpMethod`, `Content`, `McpPrompt`, `McpResource`.
- `src/registry.rs` — thread-safe tool registry / dispatch.
- `src/resources.rs` — static / file / template resource providers.
- `src/transport.rs` — `stdio` and `memory` transports.
- `src/server.rs` — MCP server routing requests to tools/resources.
- `src/client.rs` — `McpClient` for connecting to external MCP servers.
- `src/middleware.rs` — MCP middleware wired into the rvAgent pipeline.
- `src/groups.rs` — `ToolGroup`, `ToolFilter`.
- `src/topology.rs` — topology strategies for multi-agent routing.
- `src/skills_bridge.rs` — bridge for Claude Code and Codex skills formats.
- `tests/` — `integration.rs`, `stress.rs`.

## Related

`rvagent-middleware`, `rvagent-core`.
