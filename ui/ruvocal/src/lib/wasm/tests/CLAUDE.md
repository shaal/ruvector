# ui/ruvocal/src/lib/wasm/tests/

Vitest tests for the browser WASM loader.

## Files

- `wasm-capabilities.test.ts` — verifies the runtime has the WASM capabilities the `rvagent_wasm` module requires (SharedArrayBuffer, threads, etc.) and that the loader/IDB cache behave correctly.
