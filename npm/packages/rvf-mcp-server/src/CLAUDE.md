# rvf-mcp-server/src

TypeScript source for `@ruvector/rvf-mcp-server`. Compiled to `dist/` by `tsc`.

## Files

- `index.ts` — public barrel: `RvfMcpServer`, `RvfMcpServerOptions`, `createStdioServer`, `createSseServer`, `createServer`.
- `server.ts` — `RvfMcpServer` core implementation. Wraps `@modelcontextprotocol/sdk`'s `McpServer`. Defines `RvfMcpServerOptions` (name, version, defaultDimensions, maxStores) and the internal `StoreHandle` shape; registers RVF tools/resources/prompts.
- `transports.ts` — transport factories: `createStdioServer` for `StdioServerTransport`, `createSseServer` for SSE/HTTP via Express, and a unified `createServer` switch. Includes an `.mcp.json` snippet in JSDoc.
- `cli.ts` — `rvf-mcp-server` CLI binary. Parses `--transport stdio|sse` (default stdio) and `--port` (default 3100), then calls `createServer`.
