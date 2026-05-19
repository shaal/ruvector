# ui/ruvocal/chart/env/

Per-environment Helm values overlays for the `chat-ui` chart.

## Files

- `dev.yaml` — development environment overrides (lower replicas, debug config).
- `prod.yaml` — production environment overrides (HPA, ingress, resource requests).

Use with: `helm ... -f chart/env/<env>.yaml`.
