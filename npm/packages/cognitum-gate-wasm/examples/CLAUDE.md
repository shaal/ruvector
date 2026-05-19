# cognitum-gate-wasm / examples

Reference usage demos for `@cognitum/gate`. Each example exists as `.ts`/`.tsx`
source plus compiled `.js`, `.d.ts`, and source maps.

## Files
- `basic-usage.ts` - Minimal node example creating a gate and issuing
  permit/defer/deny decisions.
- `express-middleware.ts` - Wires the gate into an Express middleware to
  guard request handling.
- `react-hook.tsx` - React hook demonstrating client-side use in a UI.

These files are illustrative; they are not part of the published package
entry but are kept in-repo for documentation and copy-paste use.
