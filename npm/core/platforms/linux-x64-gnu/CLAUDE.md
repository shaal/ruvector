# npm/core/platforms/linux-x64-gnu

Published as `ruvector-core-linux-x64-gnu` (v0.1.26) - the Linux x64 (glibc) native binding for `@ruvector/core`. Restricted to `os: ["linux"]`, `cpu: ["x64"]`.

## Files

- `package.json` - CommonJS, platform-restricted, `node >= 18`.
- `index.js` - One-line require of `./ruvector.node` with a friendly error wrapper.
- `ruvector.node` - Compiled NAPI-RS shared object (~5.1 MB).

## Related

- `../../src/index.ts` - Loader that resolves to this package on linux-x64.
- `../../native/linux-x64/` - Locally-built variant (with a richer `index.cjs` shim) used during development.
- `../../../../crates/ruvector-core` - Source Rust crate.
