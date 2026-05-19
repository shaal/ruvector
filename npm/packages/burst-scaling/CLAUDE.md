# @ruvector/burst-scaling

Adaptive burst scaling system for `ruvector` services that handles 10-50x traffic spikes by combining predictive (event-driven, ML-based forecasting) and reactive (real-time metrics) scaling across GCP regions. Coordinates GCP Cloud Run / Compute Engine / Cloud SQL / Redis through Google Cloud SDKs and ships Terraform for the supporting infrastructure.

## Important files

- `package.json` — `@ruvector/burst-scaling` v1.0.0. Main `index.js`. Deps: `@google-cloud/{monitoring,compute,cloud-sql-connector,redis,logging}`, `node-cron`. Scripts: `build` (tsc), `predictor`/`scaler`/`manager` (ts-node entry points), `terraform:{init,plan,apply,destroy}`, `deploy`.
- `index.ts` — `BurstScalingSystem` orchestrator that wires together the predictor, scaler, and capacity manager; schedules metric collection and per-region orchestration. Also exports the three components.
- `burst-predictor.ts` — `BurstPredictor` and `EventCalendar` for predictive scaling based on event calendars, historical patterns, and ML forecasting.
- `reactive-scaler.ts` — `ReactiveScaler` for real-time auto-scaling driven by metrics (CPU, memory, connections) with dynamic thresholds.
- `capacity-manager.ts` — `CapacityManager` for cross-region capacity allocation, budget-aware decisions, priority allocation, traffic shedding.
- `monitoring-dashboard.json` — pre-built GCP monitoring dashboard.
- `RUNBOOK.md` — operational runbook.
- `terraform/` — Terraform IaC.

## What's exported

`BurstScalingSystem`, `BurstPredictor`, `ReactiveScaler`, `CapacityManager`, plus related types (`PredictedBurst`, `ScalingMetrics`, `ScalingAction`, `CapacityPlan`, `EventCalendar`). Compiled `.js` + `.d.ts` already present in-tree alongside `.ts` sources.

## Related

- Sibling: `npm/packages/cloud-run` (Cloud Run streaming service that this scales).
- Top-level `ruvector` package is what's being scaled.
