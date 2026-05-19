# ui/ruvocal/

SvelteKit chat application (a fork of HuggingChat / `chat-ui`, package name `chat-ui` v0.20.0, branded as **ruvocal**). It provides a multimodal LLM chat interface with MCP tool calling, an intelligent LLM router, voice transcription, image uploads, conversation sharing, OpenID auth, and a Postgres/Mongo-backed persistence layer.

## Tech stack

- **Framework:** SvelteKit 2 + Svelte 5 (runes).
- **Build:** Vite 6, TypeScript 5.5, two adapters — `@sveltejs/adapter-node` (default) and `@sveltejs/adapter-static` (set `ADAPTER=static`).
- **Styling:** TailwindCSS 3 + `@tailwindcss/typography`, PostCSS, custom CSS in `src/styles/`.
- **Backend:** SvelteKit server routes, MongoDB + Postgres, MCP SDK (`@modelcontextprotocol/sdk`), OpenAI SDK, OpenID Connect.
- **WASM:** loads `rvagent_wasm` from `static/wasm/` (provides MCP-style tools in the browser).
- **Other:** marked + KaTeX + highlight.js, satori for OG images, three.js for 3D, pino for logs, prom-client for metrics.

## Important files

- `package.json` — scripts: `dev`, `build`, `build:static`, `preview`, `check`, `lint`, `format`, `test`, `populate`, `config`, `updateLocalEnv`.
- `svelte.config.js` — adapter selection, CSP/CSRF, env loading (`.env.local`, `.env`).
- `vite.config.ts` — Vite config with `unplugin-icons`.
- `tailwind.config.cjs`, `postcss.config.js`, `tsconfig.json`, `.eslintrc.cjs`, `.prettierrc`.
- `Dockerfile`, `docker-compose.yml`, `entrypoint.sh` — container build/runtime.
- `rvf.manifest.json` — application manifest (likely for an internal "Rvector" build/deploy system).
- `PRIVACY.md`, `LICENSE` — legal.

## Directories

- `src/` — SvelteKit app: `app.html`, `hooks.server.ts`, `hooks.ts`, `lib/`, `routes/`, `styles/`.
- `chart/` — Helm chart `chat-ui` for k8s deployment (with `env/dev.yaml`, `env/prod.yaml`).
- `config/` — branding example env (`branding.env.example`).
- `docs/` — user-facing documentation (`source/`) and ADRs (`adr/`).
- `mcp-bridge/` — standalone Node/Express MCP HTTP bridge that routes tool calls to backend services.
- `models/` — placeholder for legacy model definitions (now mostly empty).
- `scripts/` — helper Node/TS scripts (`config.ts`, `populate.ts`, `updateLocalEnv.ts`, `generate-welcome.mjs`).
- `static/` — static assets (favicons, manifests, branded variants for `chatui`/`huggingchat`, WASM artifacts).
- `stub/` — local npm overrides to stub native deps (`@reflink/reflink`).

## How to run

```sh
cd ui/ruvocal
npm install
# create .env.local with OPENAI_BASE_URL / OPENAI_API_KEY
npm run dev          # dev server
npm run build        # SSR Node build
ADAPTER=static npm run build   # static SPA build
npm run test         # vitest
npm run check        # svelte-check
```

## Key conventions

- SvelteKit file-system routes under `src/routes/`. `+page.svelte` is the page, `+page.ts` is the universal load, `+server.ts` is an API endpoint, `+layout.svelte` is a shared layout. `(group)` are route groups, `[param]` are route params, `[...rest]` are rest params.
- Server-only code lives under `src/lib/server/` (never imported by client). Client/shared utilities go in `src/lib/`.
- Two API surfaces: legacy `src/routes/api/...` and the v2 surface under `src/routes/api/v2/...` (uses superjson, hono-style helpers in `src/lib/server/api/`).
- Tests use vitest (`*.spec.ts` / `*.test.ts`) with setup files in `scripts/setups/`.

## Pointers

- Architecture overview: `docs/source/developing/architecture.md`.
- ADRs for the ruvocal fork and MCP integration: `docs/adr/ADR-038-RUVOCAL-FORK.md`, `docs/adr/ADR-033-RUVECTOR-RUFLO-MCP-INTEGRATION.md`, `docs/adr/ADR-035-MCP-TOOL-GROUPS.md`.
