# ui/ruvocal/src/lib/

Code aliased as `$lib` in SvelteKit. Mix of shared client+server utilities, Svelte components, stores, and server-only modules.

## Top-level files

- `APIClient.ts` — typed client for the v2 API surface (`src/routes/api/v2`).
- `buildPrompt.ts` — assembles prompts from messages, files, and templates.
- `createShareLink.ts` — produces shareable conversation URLs.
- `switchTheme.ts` — light/dark theme toggle helper.

## Subdirectories

- `actions/` — Svelte `use:` actions (`clickOutside`, `snapScrollToBottom`).
- `components/` — Svelte UI components (general, plus `chat/`, `icons/`, `mcp/`, `players/`, `voice/`, `wasm/`).
- `constants/` — static config (MCP examples, MIME maps, pagination defaults, rvagent presets).
- `jobs/` — background job logic (`refresh-conversation-stats.ts`).
- `migrations/` — MongoDB migration framework + `routines/` (versioned migration files).
- `server/` — server-only modules (DB, auth, MCP, router, endpoints, hooks, text generation). **Do not import from `.svelte` files.**
- `stores/` — Svelte stores (settings, autopilot, MCP servers, background generations, errors, etc.).
- `types/` — shared TypeScript domain types (`Conversation`, `Message`, `Model`, `User`, `Tool`, ...).
- `utils/` — pure utility functions (markdown, hashing, MCP validation, message updates, generation state, tree helpers...).
- `wasm/` — browser-side WASM loader and IndexedDB store for `rvagent_wasm` (consumed by `static/wasm/`).
- `workers/` — Web Workers (`autopilotWorker.ts`, `detailFetchWorker.ts`, `markdownWorker.ts`).
