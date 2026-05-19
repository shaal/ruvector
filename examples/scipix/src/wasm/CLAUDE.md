# scipix/src/wasm

WASM bindings for scipix (gated by the `wasm` feature).

## Files

- `mod.rs` - Module surface.
- `api.rs` - `wasm_bindgen` exported API.
- `canvas.rs` - HTMLCanvas / ImageData interop.
- `memory.rs` - WASM memory helpers.
- `worker.rs` - Web Worker support.
- `types.rs` - Shared types.

## How to build

```bash
cd /home/user/ruvector/examples/scipix/web
./build.sh
# or directly:
wasm-pack build .. --target web --out-dir web/pkg --release -- --features wasm
```

## Related

- Web demo: `../../web/`.
- Docs: `../../docs/WASM_ARCHITECTURE.md`, `../../docs/WASM_QUICK_START.md`, `../../BUILD_WASM.md`, `../../WASM_IMPLEMENTATION_SUMMARY.md`.
