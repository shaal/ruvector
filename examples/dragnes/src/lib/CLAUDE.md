# dragnes / src / lib

Shared SvelteKit library code, importable via the `$lib` alias.

## Subdirectories
- `components/` - reusable Svelte UI components (capture, charts, overlays).
- `dragnes/` - the DrAgnes business logic module (classifier, ABCDE scoring, brain client, privacy, offline queue, etc.).

## Related
- Wired into routes under `../routes/`. Type-safe imports go via `$lib/dragnes/...`.
