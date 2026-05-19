# ruvector-robotics/src/mcp

Model Context Protocol tool registrations for agentic robotics. Self-contained — no external MCP SDK dependency; emits tool defs + JSON schemas.

## Files

- `mod.rs` — registry types (`Tool`, `ToolCategory`), JSON-schema generation, top-level registry of robotics tools (perception, behavior, planning, etc.).
- `executor.rs` — `ToolExecutor` runtime that dispatches MCP tool calls to the registered handlers.
