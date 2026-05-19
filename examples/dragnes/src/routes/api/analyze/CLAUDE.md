# dragnes / src / routes / api / analyze

`POST /api/analyze` endpoint. Accepts an image *embedding* (the raw image never leaves the client) and returns combined classification context from the brain backend, enriched with PubMed literature.

## Important files
- `+server.ts` - the SvelteKit `POST` handler. Uses `$lib/dragnes/brain-client` (`searchSimilar`, `searchLiterature`) and the `LesionClass` type from `$lib/dragnes/types`.

## Related
- Client that posts to it: `$lib/dragnes/classifier.ts` -> `$lib/dragnes/brain-client.ts`. Companion endpoints: `../similar/`, `../feedback/`, `../health/`.
