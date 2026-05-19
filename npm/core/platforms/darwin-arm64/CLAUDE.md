# npm/core/platforms/darwin-arm64

Published as `ruvector-core-darwin-arm64` - the macOS ARM64 (Apple Silicon M1/M2/M3) native binding for `@ruvector/core`. Restricted to `os: ["darwin"]`, `cpu: ["arm64"]` so npm installs it only on matching hosts.

## Files

- `package.json` - v0.1.25, CommonJS, platform-restricted, `node >= 18`.
- `index.js` - One-line require of `./ruvector.node` with a friendly error wrapper.
- `ruvector.node` - Compiled NAPI-RS dylib (~4.4 MB).

## Related

- `../../src/index.ts` - Loader that resolves to this package on darwin-arm64.
- `../../../../crates/ruvector-core` - Source Rust crate.
