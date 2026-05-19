# dragnes / src / routes / api / similar

Parent route for the similar-case lookup API. The actual endpoint lives one level deeper.

## Subdirectories
- `[id]/` - dynamic SvelteKit route exposing `GET /api/similar/[id]`; returns nearest-neighbour cases for the given lesion id from the brain backend.

## Related
- Backing client: `$lib/dragnes/brain-client.ts`. Companion endpoints: `../analyze/`, `../feedback/`, `../health/`.
