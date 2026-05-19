# pi-brain/src

TypeScript source for `@ruvector/pi-brain`. Compiled to `dist/` (ESM) and `dist/cjs/` (CJS) via two tsconfigs.

## Files

- `index.ts` — barrel re-exporting `PiBrainClient` and its public types.
- `client.ts` — `PiBrainClient` HTTP SDK targeting `https://pi.ruv.io`. Exposes share/search/get/list memory methods and `consciousness compute` (IIT4/CES/ΦID/PID/bounds/auto algorithms). Defines `ShareOptions`, `SearchOptions`, `Memory`, `ConsciousnessComputeOptions`, `ConsciousnessComputeResult`.
- `cli.ts` — `pi-brain` / `π` CLI binary. Dispatches subcommands (`health`, `share`, `search`, `get`, `list`, `status`, `mcp`) and starts MCP stdio/SSE transport for `mcp`.
- `mcp.ts` — MCP server module. Defines `TOOLS` list (`brain_share`, `brain_search`, ...) that proxy to the REST API via `PiBrainClient`.
- `assets/` — bundled static assets (image used in docs/README).
