# mcp-brain-server/cloud

GCP deployment helpers for the Cloud Run brain backend (separate from the in-crate `cloudbuild-*.yaml` files at the crate root).

## Files

- `deploy-all.sh` — orchestrates a full multi-service deploy (api + sse + worker + trainer).
- `deploy-scheduler.sh` — deploys Cloud Scheduler jobs that hit cognitive endpoints on a cadence.
- `scheduler-jobs.yaml` — scheduler job declarations (cron + target URLs).
- `setup-pubsub.sh` — provisions Pub/Sub topics/subscriptions used by the worker pipeline.
- `monitoring-dashboard.json` — Cloud Monitoring dashboard spec for the brain service.

Used in conjunction with crate-root `Dockerfile.*` and `cloudbuild-*.yaml` to ship the brain to Cloud Run.
