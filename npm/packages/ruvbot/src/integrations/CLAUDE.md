# ruvbot / src / integrations

Legacy / aggregated integrations barrel kept for backwards-
compatibility. New code should prefer `../integration/` (singular)
which is the directory referenced by the package's `exports` map.

## Files
- `index.ts` - Re-exports the providers / slack / webhooks modules
  from `../integration/`.
