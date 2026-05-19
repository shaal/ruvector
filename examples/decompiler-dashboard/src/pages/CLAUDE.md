# decompiler-dashboard/src/pages

Top-level route components rendered by `../App.tsx` via
`react-router-dom`.

## Files

- `Explorer.tsx` — landing page; shows the module tree and metrics for
  the selected version.
- `Decompiler.tsx` — interactive decompiler view: pick a module, see
  original vs. beautified vs. diff between versions.
- `RvfViewer.tsx` — RVF (RuVector File) binary inspector that uses
  `../lib/rvf-parser.ts` to render header, sections, and embedded
  vector/graph metadata (~10 KB, largest page).

## Related

- `../components/` — shared UI primitives
- `../lib/` — decompile/parse logic
