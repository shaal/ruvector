# rvf-desktop / src

Single-file source for the `ruvector` desktop binary.

## Important files
- `main.rs` - the entire app: `#[derive(Embed)]` over `../../rvf/dashboard/dist/` to bake all dashboard assets (HTML / JS / CSS / WASM solver) into the binary, a `tiny_http` server on a background thread, and a `wry::WebViewBuilder` window via `tao`.

## Run
- `cargo run -p rvf-desktop --release` (the embedded `../../rvf/dashboard/dist/` must exist - build the web dashboard first).

## Related
- Embedded asset source: `../../rvf/dashboard/`.
