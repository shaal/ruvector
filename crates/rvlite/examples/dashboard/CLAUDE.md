# rvlite/examples/dashboard

Full-featured React + Vite + TypeScript dashboard demoing RvLite: vector
search, graph visualization, supply-chain simulation, filter builder, bulk
import, SQL schema browser, vector inspector.

## Key files
- `package.json`, `vite.config.ts`, `tsconfig*.json`, `eslint.config.js`,
  `postcss.config.js`, `tailwind.config.js` - Vite + TS + Tailwind setup.
- `index.html` - HTML entry.
- `filter-helpers.ts` - top-level filter-builder helpers.
- `vector-inspector-changes.md`, `BULK_IMPORT_IMPLEMENTATION.md`,
  `FILTER_BUILDER_INTEGRATION.md`, `SQL_SCHEMA_BROWSER.md`,
  `VECTOR_INSPECTOR_IMPLEMENTATION.md`, `IMPLEMENTATION_SUMMARY.md`,
  `QUICK_START.md`, `START_HERE.md`, `SUMMARY.md`, `INDEX.md` - feature
  docs / changelogs.
- `apply-*.sh` - one-shot patch scripts for applying staged changes.

## Subdirs
- `src/` - React app source.
- `public/` - static assets served as-is.
- `docs/` - additional integration / sample docs and sample CSV/JSON.
- `scripts/` - Node debug + e2e scripts.

## Related
- `../../src/` - RvLite Rust source the dashboard exercises via WASM.
