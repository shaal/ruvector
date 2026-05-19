# edge-full/pkg/dag

WASM build of the RuVector DAG workflow-orchestration module. Used for client-side directed acyclic workflow execution.

## Files
- `ruvector_dag_wasm.js` - wasm-bindgen JS glue (ES module).
- `ruvector_dag_wasm.d.ts` - TypeScript declarations.
- `ruvector_dag_wasm_bg.wasm` - Compiled WebAssembly module.
- `ruvector_dag_wasm_bg.wasm.d.ts` - Types for the raw WASM binding.

## Import path
`import { Dag } from '@ruvector/edge-full/dag'` (resolved by `../package.json` `exports`).

## Source
Built from the `ruvector-dag` Rust crate elsewhere in the repo.
