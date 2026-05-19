# rvf-node

Node.js N-API bindings for the RuVector Format runtime. Built as `cdylib`; published via napi-rs with prebuilt binaries per platform under `npm/`.

## Layout

- `Cargo.toml` — `cdylib`. Deps: `rvf-runtime`, `rvf-types`, `napi` (with `async`), `napi-derive`, `serde_json`.
- `package.json` — npm package metadata orchestrating the per-platform binaries.
- `src/lib.rs` — `#[napi]` wrappers around `RvfStore`. Maps `rvf_runtime::filter::{FilterExpr, FilterValue}`, `rvf_runtime::options::{DistanceMetric, MetadataEntry, MetadataValue, QueryOptions, RvfOptions}`, and `rvf_types::RvfError` to JS-friendly types. Provides `insert`, `query`, `delete`, `compact`, `status` async functions.
- `npm/` — one subdir per supported triple, each shipping a prebuilt `.node` binary.

## npm/ platforms

- `darwin-arm64/` — macOS Apple Silicon
- `darwin-x64/` — macOS Intel
- `linux-arm64-gnu/` — Linux aarch64 (glibc)
- `linux-x64-gnu/` — Linux x86_64 (glibc)
- `win32-x64-msvc/` — Windows x86_64 (MSVC)

## Public API (JS)

Async `RvfStore` operations matching `../rvf-runtime` plus filter/metadata DSL.

## Related

- `../rvf-runtime` — Rust source of truth
- Sibling Node.js wrappers in the monorepo (e.g. `../../agentic-robotics-node`, `../../ruvector-tiny-dancer-node`)
