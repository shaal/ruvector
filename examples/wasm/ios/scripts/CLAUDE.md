# wasm/ios/scripts

Build automation for `ruvector-ios-wasm`.

## Files

- `build.sh` (~7 KB) - Builds the WASM module (size + speed profiles), copies into `dist/` and `swift/Resources/`, and optionally regenerates TS types.

## How to run

```bash
bash /home/user/ruvector/examples/wasm/ios/scripts/build.sh
```

## Related

- Outputs: `../dist/`, `../swift/Resources/`.
- Types: `../types/`.
