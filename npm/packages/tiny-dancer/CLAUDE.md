# @ruvector/tiny-dancer

Neural router for AI agent orchestration. FastGRNN-based intelligent
routing with circuit breaker, uncertainty estimation, optional
quantization, and hot model reload. Native Node addon (napi-rs)
wrapping the Rust `ruvector-tiny-dancer-node` crate.

## Important files
- `package.json` - npm metadata (`@ruvector/tiny-dancer` v0.1.17).
  Declares `optionalDependencies` for the per-platform binaries
  (`@ruvector/tiny-dancer-{linux,darwin,win32}-*`).
- `index.js` - Platform shim. Maps `process.platform`/`arch` to per-
  platform package + `.node` filename and loads the native module.
- `index.d.ts` - TypeScript API: `RouterConfig` (`modelPath`,
  `confidenceThreshold`, `maxUncertainty`, `enableCircuitBreaker`,
  `circuitBreakerThreshold`, `enableQuantization`, `databasePath`),
  `Candidate`, routing methods.
- `test.js` - Smoke test (`npm test`).

## Exports / entry
- `main` -> `index.js`, `types` -> `index.d.ts`. Published files:
  `index.js`, `index.d.ts`, `README.md`.

## Scripts
- `build:napi` - `napi build --platform --release --cargo-cwd
  ../../../crates/ruvector-tiny-dancer-node`.
- `test` - `node test.js`.
- `publish:platforms` - publishes per-platform binary packages.

## Related
- Rust crates: `../../../crates/ruvector-tiny-dancer-node`,
  `../../../crates/ruvector-tiny-dancer-core`,
  `../../../crates/ruvector-tiny-dancer-wasm`.
- Sibling: `../tiny-dancer-linux-arm64-gnu` (and other platform
  packages).
