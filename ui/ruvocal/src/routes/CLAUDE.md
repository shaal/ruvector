# ui/ruvocal/src/routes/

SvelteKit file-system router. Each directory is a URL segment. The chat application has these top-level surfaces: the chat page itself (`+page.svelte`), the conversation pages, an HTTP API (two versions), admin, debug, auth, settings, models, sharing, healthcheck, metrics, and privacy.

## Root-level files

- `+layout.svelte`, `+layout.ts` — root layout (loads public config, applies theme, mounts nav).
- `+page.svelte` — landing page (new chat).
- `+error.svelte` — error page rendered when a load/server function throws.

## Subdirectories

- `__debug/` — internal debugging endpoints (not for production). Currently only `openai/`.
- `admin/` — admin-only routes (export, stats).
- `api/` — legacy HTTP API surface (`/api/...`).
- `api/v2/` — modern v2 API (superjson + composable resolvers in `lib/server/api/`).
- `conversation/` — conversation viewing/editing pages + per-conversation API endpoints (message create, share, stop-generating).
- `healthcheck/` — `/healthcheck` for load balancers.
- `login/`, `logout/` — OIDC login flow (callback under `login/callback/`).
- `metrics/` — Prometheus scrape endpoint.
- `models/` — public model catalog page + per-model detail and OG thumbnail.
- `privacy/` — privacy policy page.
- `r/[id]/` — short URL redirect to a shared conversation.
- `settings/` — user settings UI (grouped under `(nav)`).

## Conventions

- File suffixes: `+page.svelte` (UI), `+page.ts` (universal load), `+page.server.ts` (server load), `+server.ts` (endpoint).
- `[param]`, `[...rest]`, `(group)` segments behave as in standard SvelteKit.
- Server-only logic lives in `src/lib/server/` and is imported into `+server.ts` / `+page.server.ts` files only.
