# edge-net/relay

Node.js WebSocket relay that fronts the edge-net swarm, plus a Brain API bridge (ADR-069 Phase 1). Deployable to Google Cloud Functions Gen2.

## Important files
- `package.json` — `@ruvector/edge-net-relay` v0.2.0; scripts: `start`, `test`, `deploy` (gcloud functions deploy).
- `Dockerfile` + `deploy.sh` — container packaging and deploy helper.
- `tests/relay.test.js` — node --test relay tests.
- `CONSUMER_FLOW_VALIDATION_REPORT.md` — consumer-flow validation results.

## Run
- Local: `npm install && npm start`.
- Tests: `npm test`.
- Deploy: `npm run deploy` (requires `gcloud` and project access).

## Tech stack
- `@google-cloud/functions-framework`, `ws`.

## Related
- Dashboard consumer: `../dashboard/src/services/relayClient.ts`.
