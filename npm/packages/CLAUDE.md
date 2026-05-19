# npm/packages/

The 59 publishable npm packages that wrap, expose, or supplement the Rust crates in `../../crates/`. Mix of WASM bundles, NAPI-RS native bindings (with per-platform sub-packages), TypeScript-only libraries, CLIs, and deployment artifacts.

## Package categories

### WASM bundles (wasm-pack output)
`acorn-wasm`, `cognitum-gate-wasm`, `graph-wasm`, `ospipe-wasm`, `rabitq-wasm`, `ruqu-wasm`, `ruvector-cnn` (WASM artifacts in-tree), `ruvector-wasm`, `ruvector-wasm-unified`, `ruvllm-wasm`, `rvf-wasm`, `rudag/pkg` + `rudag/pkg-node`. Several are publish-only placeholders whose artifacts are written at publish time from sibling Rust crates.

### NAPI-RS native bindings + per-platform packages
- **`router`** + `router-{darwin-arm64, darwin-x64, linux-arm64-gnu, linux-x64-gnu, win32-x64-msvc}`
- **`ruvllm`** + `ruvllm-{darwin-arm64, darwin-x64, linux-arm64-gnu, linux-x64-gnu, win32-x64-msvc}` + `ruvllm-cli`
- **`tiny-dancer`** + `tiny-dancer-{darwin-arm64, darwin-x64, linux-arm64-gnu, linux-x64-gnu, win32-x64-msvc}`
- **`sona`** + 7 platform sub-packages under `sona/npm/`
- Standalone NAPI: `graph-node`, `node`, `pi-brain`, `rvf-node`, `rvf-solver`

### TypeScript libraries / SDKs
`agentic-integration`, `agentic-synth` (+ `agentic-synth-examples`), `burst-scaling`, `core`, `diskann`, `graph-data-generator`, `ospipe`, `postgres-cli`, `raft`, `replication`, `rudag`, `ruvbot` (large DDD layout, 60+ subdirs), `ruvector-extensions`, `rvdna`, `rvf`, `rvf-mcp-server`, `rvlite`, `scipix`, `spiking-neural`, `ruvector`.

### CLIs / orchestrators
`cli`, `cloud-run` (Docker-deployed, no `package.json`), `postgres-cli`, `ruvllm-cli`.

## Conventions

- Each per-platform package ships only `package.json`; the `.node` binary is produced at publish time from the corresponding Rust napi crate in `../../crates/`.
- WASM-only packages ship only `package.json` here; `.wasm` artifacts are usually `.gitignore`d and built on publish.
- Most packages follow `@ruvector/*` scope, but `ruvector-extensions` and `ruvbot` are unscoped.
- `ruvllm` parent is v2.5.5 but its `optionalDependencies` pin per-platform packages at v2.0.1 (publishing version-sync gap, flagged in subdirs).
- Each package has its own `CLAUDE.md` with role, entry points, scripts, deps, and Rust-crate cross-reference.

## Notable anomalies

- `diskann/` contains a zero-byte file literally named `false` (likely a shell redirect mistake); declared `index.js` is missing.
- `rabitq-wasm/` uses `package.scoped.json` instead of `package.json`; WASM artifacts are `.gitignore`d.
- `ruvector-wasm/`, `spiking-neural/`, `agentic-integration/` declare entry points (`index.js`, `dist/index.js`, etc.) that aren't in the checkout.
- `burst-scaling/` keeps compiled `.js`/`.d.ts`/`.map` alongside `.ts` sources rather than in a `dist/`.
- `rvdna/` `package.json` has a `//optionalDependencies` comment intentionally disabling unpublished platform-binary optional deps (as of 2026-05-16).
- `ruvbot/` is by far the largest package — full DDD layout (Core / Infrastructure / Integration / Learning / Channels / Plugins / Security / Skills / Swarm / Templates / API / CLI / Utils) + 15 ADRs + bundled RVF image + GCP/Terraform deploy assets + full unit/integration/e2e tests.

## Related

- Wrapped Rust crates: `../../crates/<corresponding-name>/`.
- Sibling npm directories: `../core/` (workspace root), `../tests/` (cross-package integration), `../wasm/` (WASM workspace).
- Publishing status: see `../PUBLISHING_STATUS.md` and `../VERIFICATION_COMPLETE.md`.
