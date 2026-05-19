# ui/ruvocal/src/

SvelteKit application source.

## Files

- `app.html` — SvelteKit HTML template (head/body shell).
- `app.d.ts` — SvelteKit ambient types (`App.Locals`, `App.PageData`, etc.).
- `ambient.d.ts` — repo-wide ambient TypeScript declarations.
- `hooks.server.ts` — server hooks: auth, CSRF/origin validation, request logging, error formatting. Delegates to `lib/server/hooks/`.
- `hooks.ts` — universal hooks (e.g. client-side `handleError`).

## Subdirectories

- `lib/` — SvelteKit `$lib` aliased code. Shared utilities, components, stores, types, plus **server-only code under `lib/server/`** that must never be imported from `.svelte` client code.
- `routes/` — SvelteKit file-system router. `+page.svelte`/`+page.ts`/`+server.ts`/`+layout.svelte` conventions; `[param]`/`[...rest]`/`(group)` segments.
- `styles/` — global CSS (`main.css`, `highlight-js.css`).

## Conventions

- Use `$lib/` import alias for anything under `src/lib`.
- Server-only modules live under `src/lib/server/` and never appear in client bundles.
- Tests are colocated with code as `*.spec.ts` / `*.test.ts` and use vitest.
