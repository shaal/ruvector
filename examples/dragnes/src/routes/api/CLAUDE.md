# dragnes / src / routes / api

SvelteKit server endpoints (each subdir has a `+server.ts` exporting `GET`/`POST` handlers).

## Subdirectories
- `analyze/` - `POST /api/analyze`; takes an image embedding (never raw pixels) and returns classification context enriched with PubMed literature.
- `similar/` - `/api/similar/[id]`; nearest-neighbour case lookup over the brain backend.
- `feedback/` - user feedback submission endpoint.
- `health/` - liveness/readiness probe (used by Cloud Run / Docker).

## Related
- Backing client: `$lib/dragnes/brain-client.ts`. Container probes: `../../../cloud-run.yaml`.
