# rvdna / tests

Plain `node:test` style scripts that exercise the rvdna pure-JS
fallback against real 23andMe-format fixtures.

## Files
- `test-biomarker.js` - Default test runner used by `npm test`.
  Validates biomarker scoring against the reference ranges.
- `test-real-data.js` - Larger end-to-end run over the fixture files.
- `fixtures/` - Sample 23andMe genotype text files (see subdir).
