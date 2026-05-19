# exo-wasm/examples

## Files

- `browser_demo.html` — self-contained HTML page that loads the
  `wasm-pack`-built JS/wasm output and exercises the EXO substrate from
  the browser.

## Run

```bash
wasm-pack build .. --target web --release
# Serve the parent dir (e.g. with `python3 -m http.server`) and open
# /examples/browser_demo.html
```

## Related

- `../src/lib.rs` — bindings consumed here
- `../../../../wasm-react/` — fuller React-based example
