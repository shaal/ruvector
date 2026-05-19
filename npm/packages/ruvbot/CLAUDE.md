# ruvbot

Enterprise-grade self-learning AI assistant. Combines RuVector's HNSW
vector search, AIDefence security guards, multi-LLM provider routing
(Anthropic, OpenRouter, Google AI), Slack/Discord/webhook channels,
and an optional rvf (Rust virtual function) kernel image for sandboxed
execution. Domain-Driven Design layout with Core / Infrastructure /
Integration / Learning bounded contexts.

## Important files
- `package.json` - npm metadata (`ruvbot` v0.3.1). CLI `bin/ruvbot.js`,
  dual CJS/ESM `dist` output, subpath exports for `./core`,
  `./integrations/slack`, `./integrations/webhooks`, `./learning`.
- `Dockerfile`, `docker-compose.yml`, `.env.example` - Container/dev
  config.
- `ruvbot.rvf` - Pre-built RVF (Rust virtual function) bundle (~3.7 MB).
- `kernel/bzImage` - Linux kernel image used by the RVF sandbox runtime.
- `tsup.config.ts`, `tsconfig.json`, `tsconfig.esm.json`,
  `vitest.config.ts` - Build/test tooling.
- `bin/` - Executable entry points (`ruvbot.js`, `cli.js`).
- `scripts/` - Postinstall hook, RVF build/run helpers.
- `src/` - All TypeScript source (see subdirs below).
- `tests/` - Unit / integration / e2e suites (vitest).
- `docs/` - Architecture docs and 15 ADRs.
- `deploy/` - GCP Cloud Run + Terraform deployment assets.

## Exports / scripts
- Main `./dist/index.js` (CJS) and `./dist/esm/index.js` (ESM).
- `npm run build` -> `build:cjs` + `build:esm`; `dev` watches TS;
  `test` (vitest), `lint`, `typecheck`, `clean`.
- `build:rvf` / `run:rvf` / `inspect:rvf` manage the bundled RVF image.

## Key dependencies
- Required: `@anthropic-ai/sdk`, `aidefence`, `commander`, `chalk`,
  `ora`, `dotenv`, `eventemitter3`, `uuid`, `pino`, `pino-pretty`,
  `zod`.
- Optional: `@ruvector/ruvllm`, `@slack/bolt`, `@slack/web-api`,
  `bullmq`, `ioredis`, `pg`.

## Related
- Vector backend: `@ruvector/core`, `@ruvector/ruvllm` (sibling npm
  packages) and `../../../crates/ruvector*` crates.
- AIDefence: external `aidefence` package; see ADR-014 in `docs/adr/`.
