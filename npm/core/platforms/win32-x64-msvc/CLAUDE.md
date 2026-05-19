# npm/core/platforms/win32-x64-msvc

Published as `ruvector-core-win32-x64-msvc` - the Windows x64 (MSVC runtime) native binding for `@ruvector/core`. Restricted to `os: ["win32"]`, `cpu: ["x64"]`.

## Files

- `package.json` - v0.1.25, CommonJS, platform-restricted, `node >= 18`.
- `index.js` - Requires `./ruvector.node`, with an error message reminding users to install the Visual C++ Redistributable if loading fails.
- `ruvector.node` - Compiled NAPI-RS DLL (~6.1 MB).

## Related

- `../../src/index.ts` - Loader that resolves to this package on win32-x64.
- `../../../../crates/ruvector-core` - Source Rust crate.
