# edge-net/src/mcp

In-WASM MCP (Model Context Protocol) server.

## Important files
- `mod.rs` — module entry; exposes JS bindings.
- `protocol.rs` — MCP message types.
- `transport.rs` — transport layer (browser-friendly).
- `handlers.rs` — tool/resource handlers.

## Related
- UI: `../../dashboard/src/components/mcp/MCPTools.tsx`.
- Tests: `../../tests/mcp_integration_test.rs`.
