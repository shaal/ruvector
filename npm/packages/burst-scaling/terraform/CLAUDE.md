# burst-scaling/terraform

Terraform infrastructure-as-code for the `@ruvector/burst-scaling` system.

## Files

- `main.tf` — main Terraform module: GCP resources (Cloud Run services, monitoring, redis, etc.) needed for burst scaling.
- `variables.tf` — input variables (regions, project IDs, budget thresholds, etc.).

Driven by the parent package scripts `npm run terraform:init|plan|apply|destroy`.
