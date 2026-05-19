# decompiler-dashboard/src/lib

Pure (non-React) TypeScript helpers powering the decompiler pipeline.

## Files

- `npm-fetch.ts` — fetches RuVector tarballs from the npm registry
  (browser `fetch` -> `ArrayBuffer` -> `jszip`).
- `decompiler.ts` — orchestrates: download -> unpack -> split ->
  beautify -> hand off to the UI.
- `module-splitter.ts` — heuristics that carve a single bundled JS file
  back into per-module sources (~12 KB, largest file here).
- `beautifier.ts` — thin wrapper around `js-beautify` with the project's
  preferred options.
- `rvf-parser.ts` — parses RVF (RuVector File) binary headers / sections
  so the RvfViewer page can render structured metadata.

## Related

- `../pages/Decompiler.tsx`, `../pages/RvfViewer.tsx` — primary callers
- `../types/` — shared type definitions
