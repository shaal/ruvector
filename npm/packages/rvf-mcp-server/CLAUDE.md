# @ruvector/rvf-mcp-server

MCP (Model Context Protocol) server for the **RuVector Format (RVF)** vector database. Exposes RVF store operations as MCP tools and resources over stdio or SSE transports, so an MCP-capable client (Claude Desktop, Claude Code, etc.) can create/open/query RVF stores.

## Important files

- `package.json` — `@ruvector/rvf-mcp-server` v0.1.3. ESM. Main `dist/index.js`, types `dist/index.d.ts`. Bin: `rvf-mcp-server` → `dist/cli.js`. Deps: `@modelcontextprotocol/sdk`, `@ruvector/rvf`, `express`, `zod`. Scripts: `build` (tsc), `start` (`node dist/cli.js`), `start:stdio`, `start:sse` (port 3100), `dev` (watch).
- `src/index.ts` — barrel re-exporting `RvfMcpServer`, `RvfMcpServerOptions`, `createStdioServer`, `createSseServer`, `createServer`. JSDoc lists the registered tools (`rvf_create_store`, `rvf_open_store`, `rvf_close_store`, `rvf_ingest`, `rvf_query`, `rvf_delete`, `rvf_delete_filter`, `rvf_compact`, `rvf_status`, `rvf_list_stores`) and resources (`rvf://stores`, `rvf://stores/{storeId}/status`).
- `src/server.ts` — `RvfMcpServer` implementation. Registers tools, resources, and prompts with the MCP SDK. Manages `StoreHandle` map keyed by store ID. Options: `name`, `version`, `defaultDimensions`, `maxStores`.
- `src/transports.ts` — `createStdioServer`, `createSseServer`, `createServer` factories using `@modelcontextprotocol/sdk` transports.
- `src/cli.ts` — `#!/usr/bin/env node` CLI. Parses `--transport stdio|sse` and `--port`, then invokes the appropriate factory.

## Related

- Sibling: `@ruvector/rvf` (the underlying SDK).
- Sibling MCP server: `npm/packages/pi-brain` (also exposes an MCP server, `pi-brain mcp`).
