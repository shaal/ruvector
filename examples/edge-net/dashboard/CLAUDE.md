# edge-net/dashboard

React 19 + Vite + Tailwind monitoring/admin UI for an edge-net deployment. Visualizes nodes, credits, brain status, MCP tools, identity, economics, and WASM modules. Working demo.

## Important files
- `package.json` — `@ruvector/edge-net-dashboard` v0.1.0; scripts: `dev`, `build`, `lint`, `test`, `preview`, `docker:*`.
- `index.html` / `src/main.tsx` / `src/App.tsx` — Vite entry.
- `src/components/` — feature panels (see subdirectories).
- `src/stores/` — Zustand state slices.
- `src/services/` — `edgeNet`, `relayClient`, `storage` clients.
- `src/hooks/`, `src/types/`, `src/utils/`, `src/tests/`.
- `vite.config.ts`, `tsconfig*.json`, `tailwind.config.js`, `postcss.config.js`, `eslint.config.js`.
- `vitest.config.ts` + `e2e/` (Playwright) + `tests/` (CLI tests).
- `Dockerfile` + `docker-compose.yml` + `nginx.conf` — production container.
- `public/` — static assets (crystal.svg, vite.svg).
- `test-results/` — Playwright failure artifacts.

## Run / build
- Dev: `npm install && npm run dev`.
- Build: `npm run build`.
- Unit: `npm test`. Coverage: `npm run test:coverage`.
- E2E: `npx playwright test`.
- Docker: `npm run docker:build && npm run docker:run` (port 3000).

## Tech stack
- React 19, HeroUI, TanStack Query, Framer Motion, Recharts, Zustand, Tailwind 3.
- Vitest + Testing Library + Playwright + happy-dom/jsdom.

## Related
- Talks to `../relay/` over WebSocket; consumes WASM from `../pkg/`.
