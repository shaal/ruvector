# neural-trader/mcp

Model Context Protocol (MCP) server example exposing neural-trader to Claude Code / other MCP clients.

## Files
- `mcp-server.js` - Spins up an MCP server (via `@neural-trader/mcp`) exposing 87+ trading tools over JSON-RPC 2.0, suitable for direct integration with Claude Code.

## Run
```
npm run mcp:server
```

## Related
- Parent: `../CLAUDE.md`.
- Tools used by the server live across `../strategies/`, `../portfolio/`, `../risk/`.
