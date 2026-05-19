# edge-net/dashboard/e2e

Playwright end-to-end specs for the edge-net dashboard.

## Important files
- `dashboard.spec.ts` — navigates the running dashboard and asserts page renders / interactions.

## Run
- `npx playwright test` from `../` (configured via `../playwright.config.ts`).
- Failure artifacts land in `../test-results/`.
