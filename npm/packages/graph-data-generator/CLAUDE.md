# @ruvector/graph-data-generator

AI-powered synthetic graph data generator with OpenRouter / Kimi K2 integration, producing Neo4j knowledge graphs, social networks, temporal events, and Cypher exports. Optionally enriches nodes with vector embeddings.

## Key files

- `package.json` — `@ruvector/graph-data-generator` v0.1.0; ESM (`"type": "module"`); built with `tsup`; bin `graph-synth`.
- `tsconfig.json`, `.env.example`, `.gitignore`, `LICENSE`.

## Subdirectories

- `bin/` — `cli.js` for `graph-synth`.
- `examples/` — `basic-usage.ts` and `integration-with-agentic-synth.ts` (with compiled JS).
- `src/` — TS sources (also compiled in place):
  - `index.ts` — package barrel.
  - `cypher-generator.ts` — Cypher script emitter.
  - `embedding-enrichment.ts` — vector embedding enrichment.
  - `openrouter-client.ts` — OpenRouter API client.
  - `types.ts` — shared types.
- `src/generators/` — `entity-relationships`, `knowledge-graph`, `social-network`, `temporal-events`.
- `src/schemas/` — Zod schemas for generator inputs/outputs.

## Published API / exports

- `.` -> main barrel
- `./generators` -> all graph generators
- `./schemas` -> Zod schemas

## Scripts

- `build`, `build:generators`, `build:schemas`, `build:all` -> `tsup` (ESM+CJS+dts)
- `dev` -> tsup watch
- `test` / `test:watch` / `test:coverage` -> vitest
- `typecheck`, `lint`, `lint:fix`, `format`, `format:check`

## Key deps

`@ruvector/agentic-synth` (peer/optional), `dotenv`, `p-retry`, `p-throttle`, `zod`.

## Related

- Sibling: `npm/packages/agentic-synth*` packages.
- Sibling: `npm/packages/graph-node/` (native graph DB) and `npm/packages/ruvector/` (consumer).
