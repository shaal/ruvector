# ruvector-attention-node/npm

Per-platform NAPI-RS distribution directories for `@ruvector/attention`. Each
subdirectory is published as an optional dependency (`@ruvector/attention-<target>`)
and contains the prebuilt `.node` binary plus a minimal `package.json` pinning the
target's `os` and `cpu`.

Targets present:

- `darwin-arm64/` — Apple Silicon macOS
- `darwin-x64/` — Intel macOS
- `linux-arm64-gnu/` — Linux aarch64 (glibc)
- `linux-x64-gnu/` — Linux x86_64 (glibc)
- `linux-x64-musl/` — Linux x86_64 (musl)
- `win32-arm64-msvc/` — Windows ARM64 (MSVC)
- `win32-x64-msvc/` — Windows x64 (MSVC)

Run `npm run build` (NAPI CLI) at the crate root to populate; `npm run prepublishOnly`
to stage for publish.
