# @ruvector/pi-brain

CLI + SDK + MCP server for π — the RuVector "shared brain" hosted at `pi.ruv.io`. Lets agents share, search, and transfer learnings (architecture patterns, solutions, conventions, etc.) across Claude/MCP sessions, and exposes a small `consciousness compute` endpoint for IIT4/CES/ΦID computations.

## Important files

- `package.json` — `@ruvector/pi-brain` v0.1.2. Bin: `pi-brain` and `π` both map to `dist/cli.js`. Exports `.` (main SDK), `./mcp` (MCP server), `./client` (low-level client). Dual ESM/CJS via `tsconfig.json` + `tsconfig.cjs.json`. Dep: `@modelcontextprotocol/sdk`.
- `src/index.ts` — barrel: re-exports `PiBrainClient` and types (`ShareOptions`, `SearchOptions`, `Memory`, `ConsciousnessComputeOptions`, `ConsciousnessComputeResult`).
- `src/client.ts` — `PiBrainClient` SDK. Talks to `https://pi.ruv.io`. Defines memory share/search interfaces and the consciousness compute API.
- `src/cli.ts` — Commander-style CLI (`pi-brain health|share|search|list|status|mcp`). Spawns MCP server in stdio or SSE mode.
- `src/mcp.ts` — MCP server that proxies tools (`brain_share`, `brain_search`, etc.) to the REST API.
- `src/assets/image.png` — embedded asset.

## Scripts

`build` (ESM + CJS), `build:esm`, `build:cjs`, `start` (run CLI), `dev` (`tsc --watch`).

## Related

- Hosted backend at `pi.ruv.io` (not in this repo).
- Sibling MCP server: `npm/packages/rvf-mcp-server`.
