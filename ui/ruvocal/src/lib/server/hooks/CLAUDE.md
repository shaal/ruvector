# ui/ruvocal/src/lib/server/hooks/

Implementations behind SvelteKit's `hooks.server.ts`. Split out for clarity.

## Files

- `init.ts` — process-startup init (DB connect, migrations, exit handler registration).
- `handle.ts` — main `handle` hook: request-context setup, auth resolution, CSRF/origin enforcement, response shaping.
- `fetch.ts` — server-side `fetch` override (e.g. cookie forwarding, internal routing).
- `error.ts` — `handleError` hook: structured logging, Sentry-style reporting.
