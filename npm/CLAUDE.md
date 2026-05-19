# npm

Top-level npm workspace for the ruvector monorepo's JavaScript / TypeScript ecosystem. Hosts the public packages that wrap the Rust crates (`@ruvector/*`, the unscoped `ruvector` meta-package, CLI, etc.) as well as the per-platform native binding packages and the WASM build.

## Important files

- `package.json` - Private root workspace `@ruvector/workspace` (v0.1.0). Declares `workspaces: ["packages/*"]`, lockfile overrides for transitive vulnerabilities (`node-forge`, `flatted`, `picomatch`, `lodash`, `brace-expansion`). Requires Node >= 18, npm >= 9.
- `tsconfig.json` - Base TypeScript config (CJS, ES2020, strict, composite/incremental). Per-package configs extend or override this.
- `.eslintrc.json` - Root ESLint config using `@typescript-eslint/recommended` with type-aware rules.
- `.prettierrc.json` - Prettier formatting config.
- `.gitignore`, `.eslintrc.json`, `.prettierrc.json` - Tooling configs.
- `package-lock.json` - Workspace-wide lockfile (~1 MB).
- `PUBLISHING_STATUS.md`, `VERIFICATION_COMPLETE.md` - Release / verification notes.

## Workspace scripts

- `npm run build` - `npm run build --workspaces --if-present` (delegates to each package).
- `npm test` - Runs `tests/run-all-tests.js` (unit + integration suites).
- `npm run test:unit`, `test:integration`, `test:perf` - Filtered test runs.
- `npm run test:workspaces` - Forward `npm test` to each workspace.
- `npm run lint`, `format`, `typecheck`, `clean` - Standard workspace-wide tasks.

## Subdirectories

- `core/` - Source for `@ruvector/core` (NAPI-RS native bindings wrapper).
- `wasm/` - Source for `@ruvector/wasm` (WebAssembly bindings).
- `packages/` - Workspace member packages (declared in `package.json`).
- `tests/` - Cross-package unit, integration, and performance tests using `node:test`.

## Related

- `../crates/` - Rust crates (`ruvector-core`, `ruvector-wasm`, and many others) that these packages expose.
- `../crates/ruvector-cli/` - Rust CLI source, mirrored by an npm-distributed CLI under `packages/`.
