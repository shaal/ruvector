# ruvbot / deploy / gcp / terraform

Terraform module that provisions the GCP resources required to run
ruvbot on Cloud Run.

## Files
- `main.tf` - Defines the Cloud Run service, Artifact Registry repo,
  Secret Manager bindings, and IAM service account used by the
  Cloud Build pipeline in `../cloudbuild.yaml`.
