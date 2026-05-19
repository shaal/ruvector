# core / scripts

Build/publish helpers for `@ruvector/core`.

## Files
- `publish-platforms.js` - Iterates over the per-platform packages
  (`ruvector-core-*`) and publishes each native binary to npm. Invoked
  via `npm run publish:platforms` from the parent `package.json`.
