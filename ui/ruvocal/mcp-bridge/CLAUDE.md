# ui/ruvocal/mcp-bridge/

Standalone Node.js/Express service that acts as an **MCP (Model Context Protocol) bridge**: routes AI tool calls from the chat UI to backend services and proxies multi-provider chat traffic. Deployable independently (own `package.json`, Dockerfile, and Cloud Build config).

## Tech stack

- Node 20+, ES modules.
- Express 4 as the HTTP layer.

## Files

- `package.json` — `name: mcp-bridge` v1.0.0. Scripts: `start` (`node index.js`), `dev` (`node --watch index.js`). Note: declares `main: index.js` but the actual entrypoint here is `mcp-stdio-kernel.js`.
- `mcp-stdio-kernel.js` — MCP stdio kernel implementation; the runtime entrypoint.
- `test-harness.js` — local harness for exercising the bridge.
- `Dockerfile` — container build.
- `cloudbuild.yaml` — Google Cloud Build pipeline.

## Related

- The chat UI's server-side MCP code lives in `../src/lib/server/mcp/` and consumes/coordinates with this bridge.
- See `../docs/adr/ADR-033-RUVECTOR-RUFLO-MCP-INTEGRATION.md` and `ADR-034-OPTIONAL-MCP-BACKENDS.md`.
