# ruvector-extensions/src

TypeScript source for `ruvector-extensions`. Compiled to `dist/` via `tsc`.

## Files

- `index.ts` — barrel exporting embedding providers, graph exporters, temporal module, persistence module, and the UI server.
- `embeddings.ts` — abstract `EmbeddingProvider`, concrete `OpenAIEmbeddings`, `CohereEmbeddings`, `AnthropicEmbeddings`, `HuggingFaceEmbeddings`. Helpers `embedAndInsert` and `embedAndSearch` integrate with a `ruvector` `VectorDB`. Defines `RetryConfig`, `EmbeddingResult`, `BatchEmbeddingResult`, `EmbeddingError`, `DocumentToEmbed`, and provider config interfaces.
- `exporters.ts` — graph builders `buildGraphFromEntries` / `buildGraphFromVectorDB` plus format exporters (`exportToGraphML`, `exportToGEXF`, `exportToNeo4j`, `exportToNeo4jJSON`, `exportToD3`, `exportToD3Hierarchy`, `exportToNetworkX`, `exportToNetworkXEdgeList`, `exportToNetworkXAdjacencyList`) and a unified `exportGraph`.
- `temporal.ts` — temporal tracking, version control, change tracking, and time-travel (EventEmitter-based, uses node `crypto.createHash`).
- `persistence.ts` — JSON / binary (MessagePack) / SQLite save+load, snapshots, incremental saves, compression, progress callbacks.
- `ui-server.ts` — Express + WebSocket server exposing a graph visualization UI; uses `ruvector`'s `VectorDB` as the data source. Defines `GraphNode`, `GraphLink`.
- `examples/` — runnable example scripts (see subdir).
- `ui/` — bundled UI assets served by `ui-server.ts`.

Each `.ts` source has compiled `.js`, `.d.ts`, and `.map` siblings.
