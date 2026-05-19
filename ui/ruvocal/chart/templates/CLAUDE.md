# ui/ruvocal/chart/templates/

Helm template files rendered into Kubernetes manifests when the `chat-ui` chart is installed.

## Files

- `_helpers.tpl` — shared template helpers (naming, labels).
- `deployment.yaml` — Deployment for the SvelteKit Node server.
- `service.yaml` — ClusterIP service exposing the deployment.
- `ingress.yaml` / `ingress-internal.yaml` — public and internal ingress rules.
- `hpa.yaml` — HorizontalPodAutoscaler.
- `config.yaml` — ConfigMap with runtime env config.
- `infisical.yaml` — Infisical secret-injection integration.
- `network-policy.yaml` — NetworkPolicy rules.
- `service-account.yaml` — ServiceAccount for the workload.
- `service-monitor.yaml` — Prometheus ServiceMonitor for metrics scraping.

## Conventions

- Standard Helm conventions; values come from `../values.yaml` overlaid with `../env/*.yaml`.
