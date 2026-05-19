# rvlite/examples/dashboard/src/hooks

React hooks abstracting RvLite state and learning interactions.

## Files
- `useRvLite.ts` - load the WASM module, manage the singleton `RvLite`
  instance, expose insert / search / cypher / sparql / sql helpers, and
  trigger save/load against IndexedDB.
- `useLearning.ts` - hook driving the `NeuralEngine` (../lib) for
  the demo learning loop.
