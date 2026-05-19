# wasm/ios/swift/Resources

Bundled resources shipped with the Swift Package.

## Files

- `recommendation.wasm` (~112 KB) - WASM module loaded by `../WasmRecommendationEngine.swift` at runtime. Built with browser features (larger than the bare `../../dist/recommendation.wasm`).

## How to regenerate

```bash
cd /home/user/ruvector/examples/wasm/ios
bash scripts/build.sh
```

## Related

- Loader: `../WasmRecommendationEngine.swift`.
- Source: `../../src/`.
