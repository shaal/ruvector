# npm/core/platforms/linux-arm64-gnu

Published as `ruvector-core-linux-arm64-gnu` - the Linux ARM64 (aarch64, glibc) native binding for `@ruvector/core`. Restricted to `os: ["linux"]`, `cpu: ["arm64"]`.

## Files

- `package.json` - v0.1.25, CommonJS, platform-restricted, `node >= 18`.
- `index.js` - One-line require of `./ruvector.node` with a friendly error wrapper.
- `ruvector.node` - Compiled NAPI-RS shared object (~4.5 MB).

## Related

- `../../src/index.ts` - Loader that resolves to this package on linux-arm64.
- `../../../../crates/ruvector-core` - Source Rust crate.
