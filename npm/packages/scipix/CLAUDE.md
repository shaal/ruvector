# @ruvector/scipix

TypeScript client for the SciPix OCR service. Extracts LaTeX, MathML,
and plain text from scientific documents, equations, and technical
diagrams via an HTTP API.

## Important files
- `package.json` - npm metadata (`@ruvector/scipix` v0.1.0). Dual
  CJS/ESM export wired to `dist/index.js` / `dist/index.d.ts`.
- `src/index.ts` - Public surface: example shows `SciPixClient`,
  `ocrFile(path, { formats, detectEquations })`, `extractLatex(path)`,
  and batch processing helpers.
- `src/client.ts` - `SciPixClient` implementation. Uses Node `fs/
  promises` for file reads and a configurable `baseUrl` / `apiKey` /
  retries.
- `src/types.ts` - Types and enums (`SciPixConfig`, `OCROptions`,
  `OCRResult`, `OutputFormat`, `ImageType`, `SciPixError`,
  `SciPixErrorCode`).
- `tsconfig.json` - TS compile to `dist/`.

## Exports / entry
- `main` -> `dist/index.js`, `types` -> `dist/index.d.ts`. Published
  files: `dist`, `README.md`.

## Scripts
- `build` - `tsc`.
- `test` - `node --test test/*.test.js`.
- `typecheck`, `clean`, `prepublishOnly` (-> build).

## Related
- Backend example: `../../../examples/scipix` (per
  `homepage` in package.json).
- No direct Rust crate dep — this is an HTTP client.
