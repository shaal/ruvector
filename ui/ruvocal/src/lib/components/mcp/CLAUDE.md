# ui/ruvocal/src/lib/components/mcp/

UI for managing user-configured MCP (Model Context Protocol) servers.

## Files

- `MCPServerManager.svelte` — top-level panel listing configured servers, opens add/edit form.
- `AddServerForm.svelte` — form to add/edit an MCP server (transport URL, headers, etc.). Uses `lib/utils/mcpValidation.ts`.
- `ServerCard.svelte` — card displaying a single configured server with status / actions.

## Related

- Store: `../../stores/mcpServers.ts`.
- Server routes: `src/routes/api/mcp/servers/+server.ts`, `src/routes/api/mcp/health/+server.ts`.
- Server logic: `src/lib/server/mcp/`.
- Examples: `../../constants/mcpExamples.ts`.
