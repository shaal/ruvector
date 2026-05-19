# ui/ruvocal/src/lib/server/

Server-only modules — auth, persistence, MCP, model endpoints, the LLM router, text-generation orchestration, hooks, metrics. **Never import any module here from a `.svelte` file or other client-bundled code.** SvelteKit's bundler treats this path as server-only.

## Top-level files

- `database.ts`, `database/postgres.ts`, `database/rvf.ts` — MongoDB + Postgres clients (collections cache, pooling, helpers).
- `auth.ts` — OpenID Connect helpers, session validation.
- `adminToken.ts`, `apiToken.ts` — admin / API token validation.
- `config.ts` — server-side env/config loader.
- `conversation.ts` — conversation CRUD + tree helpers.
- `models.ts` — model registry (loaded from env / router config).
- `usageLimits.ts` — rate / quota enforcement.
- `requestContext.ts` — per-request AsyncLocalStorage context (user, request id).
- `abortRegistry.ts`, `abortedGenerations.ts` — tracking aborted generations (links to `routes/conversation/[id]/stop-generating`).
- `generateFromDefaultEndpoint.ts` — shortcut to generate against the default model endpoint (used for titles, etc.).
- `findRepoRoot.ts` — locates the repo root at runtime.
- `exitHandler.ts` — graceful-shutdown handler.
- `logger.ts` — pino logger.
- `metrics.ts` — prom-client registry / counters.
- `sendSlack.ts` — Slack webhook sender (alerts/reports).
- `isURLLocal.ts` (+ `.spec.ts`), `urlSafety.ts` — URL safety / SSRF checks.

## Subdirectories

- `__tests__/` — server-level integration tests.
- `api/` — v2 API helpers (types + auth/model/conversation resolvers + tests).
- `database/` — DB clients + tests.
- `endpoints/` — model endpoint implementations (OpenAI-compatible, image, document).
- `files/` — file upload/download.
- `fonts/` — bundled Inter TTF fonts (used by satori for OG-image generation).
- `hooks/` — SvelteKit `handle`/`fetch`/`error`/`init` hooks wiring.
- `mcp/` — MCP client pool, HF integration, tool registry.
- `router/` — intelligent LLM router (arch, policy, tools, types).
- `textGeneration/` — text-generation orchestration (streaming, reasoning, MCP flow, title).
