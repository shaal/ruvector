# dragnes / src / routes / api / health

`GET /api/health` liveness/readiness probe. Returns 200 OK with a small status payload; consumed by the Cloud Run / Docker deployment.

## Important files
- `+server.ts` - SvelteKit `GET` handler.

## Related
- Probe target referenced by `../../../cloud-run.yaml` and the `Dockerfile` HEALTHCHECK.
