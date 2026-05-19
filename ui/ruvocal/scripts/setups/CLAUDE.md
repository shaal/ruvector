# ui/ruvocal/scripts/setups/

Vitest setup files loaded before tests run.

## Files

- `vitest-setup-client.ts` — global setup for client (browser/JSDOM) tests.
- `vitest-setup-server.ts` — global setup for server-side tests (Mongo memory server, env defaults).

Referenced from `../../vite.config.ts` (`test.setupFiles`).
