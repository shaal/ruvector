# ruvector-extensions

Advanced features for the `ruvector` vector database: pluggable embedding providers, multi-format graph exporters, temporal/versioning tracking, persistence (JSON, binary, SQLite), and a small Express + WebSocket UI server. Pure TypeScript (no native bindings), depending on the `ruvector` package at runtime.

## Important files

- `package.json` — `ruvector-extensions` v0.1.0 (unscoped). Main `dist/index.js`, types `dist/index.d.ts`, ESM. Deps: `ruvector`, `@anthropic-ai/sdk`, `express`, `ws`. Optional peers: `openai`, `cohere-ai`. Scripts: `build` (tsc), `dev` (tsc --watch), `test` (`node --test`), `example:ui` (tsx).
- `src/index.ts` — barrel re-exporting everything from `embeddings`, `exporters`, `temporal`, `persistence`, `ui-server`.
- `src/embeddings.ts` — `EmbeddingProvider` base + `OpenAIEmbeddings`, `CohereEmbeddings`, `AnthropicEmbeddings`, `HuggingFaceEmbeddings` providers; helpers `embedAndInsert`, `embedAndSearch`.
- `src/exporters.ts` — graph builders (`buildGraphFromEntries`, `buildGraphFromVectorDB`) and format exporters (GraphML, GEXF, Neo4j, D3, NetworkX), plus a unified `exportGraph`.
- `src/temporal.ts` — temporal tracking / version control / time-travel for ontology + DB evolution.
- `src/persistence.ts` — save/load (JSON, binary/MessagePack, SQLite), snapshots, incremental saves, compression.
- `src/ui-server.ts` — Express + ws server exposing a graph visualization UI.
- `src/ui/` — bundled UI assets (`index.html`, `app.js`, `styles.css`).
- `src/examples/` — runnable example scripts.
- `docs/`, `examples/`, `tests/`, `EMBEDDINGS_QUICKSTART.md`, `PERSISTENCE.md`, `RELEASE_SUMMARY.md` — supporting docs and samples.

## Related

- Depends on `ruvector` (top-level npm package).
- Sibling: `npm/packages/agentic-synth` (also wraps `ruvector`).
