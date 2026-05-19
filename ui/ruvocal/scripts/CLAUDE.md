# ui/ruvocal/scripts/

Build / setup / maintenance scripts run via `vite-node` or plain Node. Wired up as npm scripts in `../package.json`.

## Files

- `config.ts` — invoked by `npm run config`; emits/validates app configuration.
- `populate.ts` — invoked by `npm run populate`; seeds the database with conversations/assistants for development. See `samples.txt`.
- `updateLocalEnv.ts` — invoked by `npm run updateLocalEnv`; refreshes `.env.local` with sensible defaults.
- `generate-welcome.mjs` — generates the welcome assets (e.g. `../static/chatui/welcome.svg`).
- `samples.txt` — sample data used by `populate.ts`.

## Subdirectories

- `setups/` — vitest setup files (`vitest-setup-client.ts`, `vitest-setup-server.ts`) referenced from `../vite.config.ts`.
