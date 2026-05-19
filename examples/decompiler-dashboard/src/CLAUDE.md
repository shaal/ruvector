# decompiler-dashboard/src

React + TypeScript source root for the decompiler dashboard SPA.

## Files

- `main.tsx` — Vite entry; mounts `<App />` with `BrowserRouter`.
- `App.tsx` — top-level layout, version selector, and routes for
  Explorer, Decompiler, and RvfViewer pages. Fetches per-version data
  from `/data/{version}/{source/metrics.json, README.md, manifest.json}`.
- `index.css` — global Tailwind directives + dark-theme base styles.
- `vite-env.d.ts` — Vite ambient types.

## Subdirectories

- `components/` — reusable UI components (search bar, version selector,
  code viewer, diff viewer, download menu, metrics card, module tree).
- `lib/` — non-UI logic: tarball/decompile pipeline, RVF parser,
  module splitter, npm fetcher, beautifier.
- `pages/` — route components: `Explorer`, `Decompiler`, `RvfViewer`.
- `types/` — shared TypeScript types (`VersionData`, `PageId`, etc.).

## Related

- `../package.json` — build scripts and dependencies
- `../vite.config.ts` — dev server config
