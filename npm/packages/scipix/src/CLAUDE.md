# scipix / src

TypeScript source for `@ruvector/scipix`.

## Files
- `index.ts` - Public barrel. Re-exports the `SciPixClient` and all
  types/enums from `client.ts` and `types.ts`.
- `client.ts` - `SciPixClient` class implementing OCR / batch /
  health methods against the SciPix HTTP API. Reads images via
  `node:fs/promises`, normalizes extensions via `node:path`, applies a
  `DEFAULT_CONFIG` with `baseUrl: 'http://localhost:8080'`, 30 s
  timeout, 3 retries, and the LaTeX + Text formats by default.
- `types.ts` - All shared types and enums: `SciPixConfig`,
  `OCROptions`, `OCRResult`, `BatchOCRRequest`, `BatchOCRResult`,
  `HealthStatus`, `SciPixError`, `SciPixErrorCode`, `OutputFormat`,
  `ImageType`.
