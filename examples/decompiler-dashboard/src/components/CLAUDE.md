# decompiler-dashboard/src/components

Reusable React components for the dashboard UI. Stateless or
minimally-stateful; all data is passed in via props from `../pages/`.

## Files

- `CodeViewer.tsx` — highlighted source viewer (uses `highlight.js`).
- `DiffViewer.tsx` — text diff renderer (uses the `diff` npm package).
- `DownloadMenu.tsx` — dropdown for downloading source / split modules
  / RVF artifacts as a `.zip` (via `jszip`).
- `MetricsCard.tsx` — single-version metrics summary card.
- `ModuleTree.tsx` — collapsible tree of decompiled modules.
- `SearchBar.tsx` — fuzzy search across modules/files.
- `VersionSelector.tsx` — dropdown of available release versions.

## Related

- `../pages/` — consumers of these components
- `../lib/` — business logic powering them (decompile, diff, beautify)
