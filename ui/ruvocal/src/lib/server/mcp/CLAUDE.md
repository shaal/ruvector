# ui/ruvocal/src/lib/server/mcp/

Server-side **MCP (Model Context Protocol)** integration: a pooled client per configured server, an HTTP transport helper, an HF-specific helper, plus a registry/tool layer that the text-generation flow consumes.

## Files

- `clientPool.ts` — caches and reuses MCP client connections per-server (avoids reconnect-per-request).
- `httpClient.ts` — HTTP MCP transport client (calls the MCP bridge or remote MCP endpoints).
- `hf.ts` — HuggingFace-specific MCP helper (auth headers, well-known endpoints).
- `registry.ts` — tracks available MCP servers and their advertised tools.
- `tools.ts` — exposes tools to the text-generation flow (`../textGeneration/mcp/`).

## Related

- HTTP API to manage user MCP servers: `src/routes/api/mcp/servers/+server.ts`, `src/routes/api/mcp/health/+server.ts`.
- Client UI: `lib/components/mcp/`.
- ADRs: `docs/adr/ADR-033-...`, `ADR-034-OPTIONAL-MCP-BACKENDS.md`, `ADR-035-MCP-TOOL-GROUPS.md`.
- Companion service: `../../../mcp-bridge/`.
