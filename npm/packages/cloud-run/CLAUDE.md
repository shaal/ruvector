# cloud-run

Cloud Run streaming service for `ruvector` — a Fastify-based HTTP/2 + WebSocket server tuned for very high concurrent connections, with a circuit-breaking load balancer, pooled vector-DB client, and OpenTelemetry / Prometheus metrics. This directory ships the Dockerfile and source for the Cloud Run deployment; there is no `package.json` here (it's built standalone via `Dockerfile`/`cloudbuild.yaml`).

## Important files

- `Dockerfile` — container image for Cloud Run.
- `cloudbuild.yaml` — Google Cloud Build pipeline.
- `streaming-service.ts` — main Fastify server entry. Configures HTTP/2 + WebSocket + compress + helmet + rate-limit; exposes `/health`, streaming, and query endpoints. Built `.js` and `.d.ts` present.
- `streaming-service-optimized.ts` — optimized variant of the streaming service.
- `vector-client.ts` — pooled, cached client wrapper around `ruvector` with OTel + Prometheus instrumentation.
- `load-balancer.ts` — internal load balancer with circuit breaker, per-client rate limiting, regional routing, and health-based routing.
- `COST_OPTIMIZATIONS.md`, `QUERY_OPTIMIZATIONS.md` — operations notes.

All `.ts` files have `.js`, `.d.ts`, and `.map` siblings already built in-tree.

## Related

- Scaled by `npm/packages/burst-scaling`.
- Talks to the `ruvector` vector DB.
