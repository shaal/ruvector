# dragnes / src / routes

SvelteKit route tree. File-system routing - each `+page.svelte` is a page, each `+server.ts` under `api/` is an HTTP endpoint.

## Important files
- `+layout.svelte` - root layout (header, global providers).
- `+page.svelte` - main DrAgnes UI page (camera + classifier + results).
- `api/` - server-side HTTP endpoints (analyze, similar, feedback, health).

## Related
- Page logic: `../lib/dragnes/` (classifier + brain client). UI components: `../lib/components/`.
