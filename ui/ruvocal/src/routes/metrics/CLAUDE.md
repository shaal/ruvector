# ui/ruvocal/src/routes/metrics/

Prometheus metrics scrape endpoint.

## Files

- `+server.ts` — `GET /metrics` returns the `prom-client` registry from `lib/server/metrics.ts` in the Prometheus text exposition format. Scraped by the `ServiceMonitor` template at `chart/templates/service-monitor.yaml`.

## Related

- Docs: `docs/source/configuration/metrics.md`.
