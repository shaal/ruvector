# nodejs

Two minimal Node.js scripts demonstrating the `ruvector` npm package:
basic insert/search and a tiny semantic-search workflow with a mock
embedding function. CommonJS (`require('ruvector')`); no
`package.json` here — you need `ruvector` installed in a parent
`node_modules/` or globally.

## Files

- `basic_usage.js` — creates a 128-d `VectorDB`, inserts one vector then
  100 vectors via `insertBatch`, queries top-K with `search`. Writes
  `examples_basic_node.db`.
- `semantic_search.js` — builds a 384-d index, indexes a handful of
  text documents through `mockEmbedding(text)`, runs example queries.

## Run

```bash
# from a directory that has `ruvector` installed:
node examples/nodejs/basic_usage.js
node examples/nodejs/semantic_search.js
```

## Tech stack

- Node.js, CommonJS `require`
- `ruvector` npm package (Node binding for the RuVector vector DB)

## Related

- `../wasm-react/` — browser-side WASM React example
- `../decompiler-dashboard/` — TS / Vite SPA
- `../exo-ai-2025/crates/exo-node/` — NAPI-RS binding crate (separate
  artifact)
