# rvdna / src

Pure-JavaScript fallback implementations used when the native
`@ruvector/rvdna-*` binary cannot be loaded. Mirrors a subset of the
Rust crate so the package degrades gracefully on unsupported
platforms.

## Files
- `biomarker.js` - Clinical reference ranges (`BIOMARKER_REFERENCES`,
  frozen array of lipid / metabolic / inflammatory / thyroid / iron
  markers) and pure-JS biomarker scoring functions.
- `stream.js` - Streaming anomaly-detection helpers (sliding window
  statistics) for time-series biosignals.
