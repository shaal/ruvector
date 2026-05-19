# dragnes / src

SvelteKit source root. Splits into shared library code (`lib/`) and routes / API endpoints (`routes/`).

## Important files
- `app.html` - HTML shell rendered by SvelteKit.
- `app.css` - global stylesheet (Tailwind directives).
- `lib/` - reusable DrAgnes module (`lib/dragnes/`) and Svelte components (`lib/components/`). See subdir CLAUDE.md files.
- `routes/` - SvelteKit pages (`+page.svelte`, `+layout.svelte`) and HTTP endpoints under `routes/api/`.

## Related
- Path alias: `$lib -> src/lib` (set in `../svelte.config.js`).
- Config: `../dragnes.config.ts`.
