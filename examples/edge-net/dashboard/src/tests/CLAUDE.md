# dashboard/src/tests

Vitest unit tests for the dashboard.

## Important files
- `setup.ts` — global test setup (jest-dom matchers, mocks).
- `App.test.tsx` — top-level render smoke test.
- `components.test.tsx` — component-level tests.
- `stores.test.ts` — Zustand store behavior.
- `debug.test.ts` — debug utility tests.

## Run
- `npm test` from `../../` (or `npm run test:watch`).
