# ui/ruvocal/src/lib/wasm/

Browser-side loader for the `rvagent_wasm` WebAssembly module that provides MCP-style tools in the browser. Pairs with `static/wasm/rvagent_wasm.js` and `rvagent_wasm_bg.wasm`.

## Files

- `index.ts` — initializes / lazy-loads the WASM module, exposes its API to other modules (used by `lib/server/textGeneration/mcp/wasmTools.test.ts` and `lib/components/wasm/`).
- `idb.ts` — IndexedDB helpers to persist WASM-tool state across sessions.

## Subdirectories

- `tests/` — WASM capability tests.

## Related

- Store: `lib/stores/wasmMcp.ts`.
- UI: `lib/components/wasm/GalleryPanel.svelte`.
