# npm/core/platforms

Per-platform sibling npm packages that ship the prebuilt NAPI-RS `.node` binary for `@ruvector/core`. Each subdirectory is published as its own package (e.g. `ruvector-core-linux-x64-gnu`); the main `@ruvector/core` loader picks the right one for the host at runtime via `optionalDependencies`-style resolution.

## Subdirectories

- `darwin-arm64/` - macOS Apple Silicon (M1/M2/M3).
- `darwin-x64/` - macOS Intel.
- `linux-arm64-gnu/` - Linux ARM64 (glibc).
- `linux-x64-gnu/` - Linux x64 (glibc).
- `win32-x64-msvc/` - Windows x64 (MSVC runtime).

Each subdir contains:
- `package.json` - Per-platform package metadata (`os`, `cpu` restrictions; `"type": "commonjs"`).
- `index.js` - Tiny shim that `require('./ruvector.node')` with a helpful error message on failure.
- `ruvector.node` - Compiled NAPI-RS dynamic library for that target.

## Related

- `../src/index.ts` - Loader that selects the appropriate package at runtime based on `os.platform()` + `os.arch()`.
- `../native/` - Locally-built (unpublished) variants used for development.
