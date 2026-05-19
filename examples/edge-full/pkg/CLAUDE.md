# edge-full/pkg

The npm-publishable package `@ruvector/edge-full` (v0.1.0): a single bundle exporting every RuVector WASM module for the browser / edge.

## Key files
- `package.json` - ES-module package metadata. Declares `exports` for the root and each sub-module (`./edge`, `./graph`, `./rvlite`, `./sona`, `./dag`, `./onnx`).
- `index.js` - Unified entrypoint. Re-exports each WASM module under namespaces and provides `initAll()` for bulk init.
- `index.d.ts` - TypeScript declarations for the unified entrypoint.
- `generator.html` - Static HTML demo / generator that exercises the bundle in a browser.
- `LICENSE` - MIT.

## Submodule directories
- `edge/` - Cryptographic identity, P2P, vector search (HNSW), neural networks.
- `graph/` - Neo4j-style graph DB with Cypher queries.
- `rvlite/` - SQL/SPARQL/Cypher vector database.
- `sona/` - Self-optimizing neural architecture with LoRA.
- `dag/` - DAG workflow orchestration.
- `onnx/` - ONNX inference with HuggingFace embedding models.

## How to use
```js
import { initAll, edge, graph, rvlite, sona, dag } from '@ruvector/edge-full';
await initAll();
```
Or import a single submodule, e.g. `import { WasmHnswIndex } from '@ruvector/edge-full/edge'`.

## How to view the demo
Serve the directory and open `generator.html`:
```
npx http-server .
```

## Tech stack
- WASM (Rust compiled via wasm-bindgen), ES modules, no runtime deps.
