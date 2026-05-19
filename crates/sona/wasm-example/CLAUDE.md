# sona/wasm-example

Browser demo for the WASM build of `ruvector-sona`.

## Files

- `index.html` — Loads the wasm-pack output and exercises the engine in-browser.
- `package.json` — Demo packaging.

## Build + serve

```
wasm-pack build crates/sona --target web --features wasm
cd crates/sona/wasm-example && npx serve
```
