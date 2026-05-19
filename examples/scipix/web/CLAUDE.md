# scipix/web

Browser harness and TypeScript types for the scipix WASM build (`@ruvector/mathpix-wasm`).

## Files

- `package.json` - npm scripts: `build` (wasm-pack), `build:dev`, `serve` (python http.server :8080), `dev`.
- `tsconfig.json` - TS config.
- `types.ts` - Hand-written TS declarations for the WASM API.
- `worker.js` - Web Worker entry that loads the WASM.
- `example.html` (~18 KB) - Standalone in-browser demo page.
- `build.sh` - Builds the WASM with `--features wasm` into `web/pkg/`.
- `.gitignore` - Ignores the generated `pkg/` directory.

## How to run

```bash
cd /home/user/ruvector/examples/scipix/web
./build.sh        # or: npm run build
npm run serve     # http://localhost:8080
```

## Tech stack

- wasm-pack + TypeScript + plain ES modules / Web Workers.

## Related

- Rust WASM source: `../src/wasm/`.
- Docs: `../docs/WASM_QUICK_START.md`, `../docs/WASM_ARCHITECTURE.md`, `../BUILD_WASM.md`.
