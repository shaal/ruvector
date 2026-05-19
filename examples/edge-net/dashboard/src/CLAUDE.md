# edge-net/dashboard/src

React/TypeScript source for the edge-net dashboard.

## Important files
- `main.tsx` — React 19 root mount.
- `App.tsx` — top-level layout, routing, theme.
- `index.css` — Tailwind base / global styles.
- `assets/` — bundled images (react.svg).
- `components/` — feature panels grouped by domain (brain, cdn, common, dashboard, docs, economics, identity, mcp, network, rewards, wasm).
- `hooks/` — shared React hooks (e.g. `useMediaQuery`).
- `services/` — backend clients (`edgeNet.ts`, `relayClient.ts`, `storage.ts`).
- `stores/` — Zustand state slices (cdn, identity, mcp, network, wasm).
- `tests/` — Vitest unit tests + setup.
- `types/index.ts` — shared type definitions.
- `utils/debug.ts` — debug helpers.

## Build
- From `../`: `npm run dev` or `npm run build`.
