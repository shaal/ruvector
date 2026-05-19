# npm/

Per-platform NAPI sub-packages for `@ruvector/graph-node`. Each subdirectory is its own publishable npm package containing a single `package.json` (the actual `.node` binary is dropped here at build/publish time via `napi build`).

Subdirectories:

- `darwin-arm64/` — `@ruvector/graph-node-darwin-arm64`
- `darwin-x64/` — `@ruvector/graph-node-darwin-x64`
- `linux-arm64-gnu/` — `@ruvector/graph-node-linux-arm64-gnu`
- `win32-x64-msvc/` — `@ruvector/graph-node-win32-x64-msvc`

The corresponding `linux-x64-gnu` package is listed as an optional dep in the parent `package.json` but its subdir isn't present in this checkout.
