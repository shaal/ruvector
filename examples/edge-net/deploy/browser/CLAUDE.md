# edge-net/deploy/browser

Embeddable browser snippet that boots edge-net inside a third-party site.

## Important files
- `embed-snippet.js` — `<script>`-able loader that pulls the WASM bundle from `../../pkg/` and starts the contributor flow.
- `example.html` — minimal page showing how to embed the snippet.

## Run
- Serve this directory with any static HTTP server (e.g. `python -m http.server`) and open `example.html`.
