# @ruvector/agentic-integration

Distributed agent coordination layer for ruvector with claude-flow integration. Provides regional agents, swarm management, and a coordination protocol for orchestrating vector-database operations across multiple regions/clouds.

## Layout

This directory contains source `.ts` files alongside compiled `.js` / `.d.ts` outputs (no separate `src/`/`dist/` split — TypeScript sources sit beside their build artifacts).

- `package.json` — `@ruvector/agentic-integration` v1.0.0; main `dist/index.js` (note: build output path differs from layout).
- `agent-coordinator.ts/.js` — top-level coordinator that schedules tasks across regional agents.
- `regional-agent.ts/.js` — per-region agent worker, exposed via the `./agent` subpath export.
- `swarm-manager.ts/.js` — mesh/swarm topology manager exposed via `./swarm`.
- `coordination-protocol.ts/.js` — message protocol / contracts exposed via `./protocol`.
- `integration-tests.ts/.js` — runnable integration test suite (`npm run test:integration`).

## Published API / exports

- `.` -> coordinator/agent/swarm/protocol re-exports
- `./coordinator`, `./agent`, `./swarm`, `./protocol`

## Scripts

- `build` -> `tsc`
- `test` -> `jest --coverage`; `test:integration` filters to `integration-tests`
- `deploy:us-east|us-west|eu-west|asia-east|all` -> `gcloud run deploy` for each region
- `swarm:init`, `swarm:status` -> claude-flow hook entry points

## Key deps

`claude-flow ^2`, `@google-cloud/pubsub`, `@google-cloud/storage`, `@grpc/grpc-js`, `fastify`, `express`, `ioredis`, `pg`, `winston`, `pino`, `zod`.

## Related

- Designed to wrap and distribute `ruvector` (parent package) workloads.
- Coordinates with claude-flow CLI hooks; no direct Rust crate.
