# @cognitum/gate (cognitum-gate-wasm)

Browser and Node.js coherence gate for AI agent safety. Delivers real-time
permit/defer/deny decisions in microseconds backed by a WASM kernel,
turning attention into a permission system rather than a popularity contest.

## Important files
- `package.json` - npm metadata (`@cognitum/gate` v0.1.0), multi-entry exports
  (`.`, `./node`, `./sw`, `./wasm`, `./experimental`).
- `src/index.ts` - TypeScript surface: `Verdict`, `Priority`, `GateConfig`,
  `TileTopology`, `ReceiptStore`, and the main gate API.
- `examples/` - usage demos (`basic-usage.ts`, `express-middleware.ts`,
  `react-hook.tsx`) shipped as TS + compiled JS.

## Exports
- Main: `./dist/{esm,cjs}/index.js` with types from `./dist/types/index.d.ts`.
- Subpath exports for Node-only (`./node`), service worker (`./sw`),
  raw WASM glue (`./wasm`), and experimental APIs.
- `sideEffects: false`, peer-deps on `claude-flow` (optional).

## Scripts (package.json)
- `build` - `build:wasm` (wasm-pack on `../cognitum-gate-kernel`) then `build:ts` (tsup).
- `test` / `test:watch` / `test:coverage` / `test:browser` - vitest based.
- `lint`, `typecheck`, `clean`, `prepublishOnly` (clean + build + test).

## Key dependencies
- Runtime: `@noble/hashes` (receipt hashing).
- Dev: tsup, vitest, eslint, typescript, jsdom.

## Related
- Rust kernel: `../../../crates/cognitum-gate-kernel` (wasm-pack source).
- Sibling tile crate: `../../../crates/cognitum-gate-tilezero`.
