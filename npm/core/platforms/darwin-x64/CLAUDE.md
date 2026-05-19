# npm/core/platforms/darwin-x64

Published as `ruvector-core-darwin-x64` - the macOS Intel native binding for `@ruvector/core`. Restricted to `os: ["darwin"]`, `cpu: ["x64"]`.

## Files

- `package.json` - v0.1.25, CommonJS, platform-restricted, `node >= 18`.
- `index.js` - One-line require of `./ruvector.node` with a friendly error wrapper.
- `ruvector.node` - Compiled NAPI-RS dylib (~4.8 MB).

## Related

- `../../src/index.ts` - Loader that resolves to this package on darwin-x64.
- `../../../../crates/ruvector-core` - Source Rust crate.
