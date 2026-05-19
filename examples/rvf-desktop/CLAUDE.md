# rvf-desktop

RuVector Causal Atlas - a native desktop app that embeds the entire `rvf/dashboard/` (Vite-built React/Three.js dashboard, including the WASM solver) at compile time via `rust-embed`, serves it from a tiny background HTTP server, and opens it in a native webview via `wry` + `tao`. Result: a single binary, no external dependencies, called `ruvector`.

## Important files
- `Cargo.toml` - declares the `ruvector` binary. Deps: `wry 0.49`, `tao 0.32`, `rust-embed 8`, `tiny_http 0.12`, `mime_guess 2`, `open 5`. Release profile is size-optimised (`opt-level = "z"`, LTO, strip).
- `Cargo.lock` - committed.
- `src/main.rs` - the whole app (~tens of lines): `Embed` struct rooted at `../../rvf/dashboard/dist/`, picks a free `TcpListener`, spawns the HTTP server on a background thread, opens a webview pointed at it.

## Build / run
- The bundled dashboard must exist at `../rvf/dashboard/dist/` (run its own build first).
- `cargo run -p rvf-desktop --release` launches the desktop app.

## Related
- Web counterpart of the dashboard: `../rvf/` (sibling, not in this chunk). Embedded asset folder it points at: `../rvf/dashboard/dist/`.
