# ruvbot / deploy / gcp

Google Cloud Platform deployment recipes for ruvbot (ADR-013).

## Files
- `cloudbuild.yaml` - Cloud Build pipeline: builds the container,
  pushes to Artifact Registry, deploys to Cloud Run.
- `deploy.sh` - Shell helper wrapping `gcloud run deploy` with the
  expected env vars / service account flags.
- `terraform/` - Infrastructure-as-code (see subdir).
