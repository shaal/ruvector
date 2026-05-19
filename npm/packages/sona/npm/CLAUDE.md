# sona/npm

Per-platform native binary subpackages for `@ruvector/sona`. Each subdirectory is its own publishable npm package (`@ruvector/sona-<platform>`), listed as an `optionalDependency` of the parent `@ruvector/sona` so npm installs only the binary matching the user's OS + CPU + libc.

## Subdirectories

- `darwin-arm64/` — `@ruvector/sona-darwin-arm64` (Apple Silicon).
- `darwin-x64/` — `@ruvector/sona-darwin-x64` (Intel Mac).
- `linux-arm64-gnu/` — `@ruvector/sona-linux-arm64-gnu`.
- `linux-x64-gnu/` — `@ruvector/sona-linux-x64-gnu`.
- `linux-x64-musl/` — `@ruvector/sona-linux-x64-musl`.
- `win32-arm64-msvc/` — `@ruvector/sona-win32-arm64-msvc`.
- `win32-x64-msvc/` — `@ruvector/sona-win32-x64-msvc`.

Each contains a `package.json` declaring `os`/`cpu`/(`libc`) constraints and shipping a single `sona.<triple>.node` binary, produced by `napi build`.
