# ui/ruvocal/src/routes/api/mcp/health/

Health-check endpoint for configured MCP servers.

## Files

- `+server.ts` — `GET` returns per-server health/status (reachability, advertised tools count). Backed by `lib/server/mcp/clientPool.ts` and `registry.ts`.
