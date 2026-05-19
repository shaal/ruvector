# ui/ruvocal/src/routes/api/mcp/servers/

CRUD endpoint for the authenticated user's MCP server configurations.

## Files

- `+server.ts` — `GET` list, `POST` create, `PATCH` update, `DELETE` remove an MCP server config. Validates payloads with `lib/utils/mcpValidation.ts`; persists via the MongoDB user/settings collections.

## Related UI

- `lib/components/mcp/MCPServerManager.svelte`, `AddServerForm.svelte`, `ServerCard.svelte`.
- Store: `lib/stores/mcpServers.ts`.
