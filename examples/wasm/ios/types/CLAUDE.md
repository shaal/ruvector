# wasm/ios/types

`@ruvector/ios-wasm-types` npm package: TypeScript declarations for the `ruvector-ios-wasm` WASM module so JS/TS consumers (web/Node) get type safety.

## Files

- `package.json` - npm manifest (`types: "ruvector-ios.d.ts"`, ships only the `.d.ts`).
- `ruvector-ios.d.ts` (~16 KB) - TS declarations describing the exported WASM API surface.

## How to publish

```bash
cd /home/user/ruvector/examples/wasm/ios/types
npm publish --access public
```

## Related

- WASM source: `../src/`.
- Build script that updates types: `../scripts/build.sh`.
