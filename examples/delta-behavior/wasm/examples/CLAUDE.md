# delta-behavior / wasm / examples

Runnable examples of the `@ruvector/delta-behavior` SDK in both browser and Node environments.

## Important files
- `browser-example.html` - drop-in HTML page that loads the built WASM + JS bundle and exercises the SDK from the browser.
- `node-example.ts` - Node.js / tsx example; run with `npm run example:node` (from `../`).

## Run
- Browser: build with `npm run build:all` (from `../`), then serve this directory and open `browser-example.html`.
- Node: `cd .. && npm run example:node`.

## Related
- SDK source: `../src/index.ts`. Dist bundle: `../dist/`.
