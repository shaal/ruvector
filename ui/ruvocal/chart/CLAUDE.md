# ui/ruvocal/chart/

Helm chart for deploying the ruvocal (chat-ui) SvelteKit app to Kubernetes.

## Files

- `Chart.yaml` — chart metadata (`name: chat-ui`, `version: 0.0.1-latest`, application type).
- `values.yaml` — default values consumed by templates.

## Subdirectories

- `env/` — per-environment value overlays (`dev.yaml`, `prod.yaml`).
- `templates/` — Helm templates rendered into k8s manifests (deployment, service, ingress, HPA, etc.).

## Usage

```sh
helm upgrade --install ruvocal ./chart -f chart/env/prod.yaml
```
