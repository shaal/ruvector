# dragnes / src / routes / api / similar / [id]

SvelteKit dynamic route. Exposes `GET /api/similar/:id` - returns similar lesion cases for the given id by querying the brain backend (vector + graph store).

## Important files
- `+server.ts` - SvelteKit `GET` handler. Uses the `searchSimilar` helper from `$lib/dragnes/brain-client`.

## Related
- Parent: `../`. Companion endpoints: `../../analyze/`, `../../feedback/`, `../../health/`.
