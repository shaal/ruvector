# neural-trader-wasm/tests

WASM smoke tests run from a JS host.

## Files
- `node-smoke.mjs` - Node ESM script that imports the built WASM package
  (post `wasm-pack build`), exercises `init` / `version` / `healthCheck`, and
  walks a small event-to-gate-to-replay flow. Used by `Dockerfile.test`.
