# src/

TypeScript source for `@ruvector/graph-data-generator`. Compiled via `tsup` into `../dist/`.

- `index.ts` — package barrel; re-exports generators, schemas, Cypher emitter, embedding enrichment.
- `cypher-generator.ts` — emits Cypher / Neo4j scripts from generated graphs.
- `embedding-enrichment.ts` — attaches vector embeddings to nodes/edges.
- `openrouter-client.ts` — OpenRouter (Kimi K2) HTTP client.
- `types.ts` — shared TypeScript types.
- `generators/` — graph-domain generators (knowledge graph, social network, temporal events, entity relationships).
- `schemas/` — Zod schemas for inputs/outputs.
