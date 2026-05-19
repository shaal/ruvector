# dragnes

DrAgnes - AI-powered dermatology intelligence platform. SvelteKit app that runs browser-based skin-lesion classification (MobileNetV3 WASM via `@ruvector/cnn`) with ABCDE dermoscopic scoring, Grad-CAM overlays, federated brain sync, offline queue, and privacy-preserving telemetry. Deployable as a Node server or Docker container (Cloud Run).

## Important files
- `package.json` - SvelteKit 2 + Svelte 5 + Vite 6 + TypeScript + Tailwind. Scripts: `dev`, `build`, `preview`, `test` (Vitest), `check`, `deploy`.
- `svelte.config.js` - uses `@sveltejs/adapter-node`, exposes `$lib` alias.
- `dragnes.config.ts` - central DrAgnes config (class taxonomy, privacy, brain sync, performance budgets).
- `vite.config.ts`, `tsconfig.json`, `tailwind.config.cjs`, `postcss.config.js` - standard SvelteKit toolchain.
- `Dockerfile`, `cloud-run.yaml` - container packaging + Google Cloud Run deployment manifest.
- `src/`, `static/`, `tests/`, `docs/`, `scripts/` - app code, static assets, tests, docs, deploy scripts (each has its own CLAUDE.md).
- `.svelte-kit/` - generated SvelteKit cache (not documented).

## Build / run
- `npm install && npm run dev` for local dev.
- `npm run build && node build` for production.
- `bash scripts/deploy.sh` (Cloud Run).
- `npm test` for Vitest.

## Related
- Sibling JS/TS / web example: none on the same scale - this is the main SvelteKit demo. Shares the "brain client" idea with `../OSpipe/` (vector + graph backend).
