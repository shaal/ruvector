# ui/ruvocal/src/routes/healthcheck/

Health check endpoint for load balancers / k8s probes.

## Files

- `+server.ts` — `GET /healthcheck` returns 200 when the app is up. Wired into the Helm chart (`chart/templates/deployment.yaml`).
