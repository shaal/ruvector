# wasm/ios/swift

Swift Package that consumes the `ruvector-ios-wasm` artifact on iOS / macOS via a WASM runtime.

## Files

- `Package.swift` - SwiftPM manifest.
- `RuvectorWasm.swift` (~19 KB) - Low-level Swift bridge to the WASM module.
- `WasmRecommendationEngine.swift` (~14 KB) - Engine wrapper that loads `Resources/recommendation.wasm`.
- `HybridRecommendationService.swift` (~11 KB) - Higher-level service combining the WASM engine with native iOS data sources.
- `Resources/recommendation.wasm` (~112 KB) - Bundled WASM (larger build than `../dist/`, includes browser features).
- `Tests/RecommendationTests.swift` - XCTest cases.

## How to build/test

```bash
cd /home/user/ruvector/examples/wasm/ios/swift
swift build
swift test
```

## Related

- WASM source: `../src/`.
- TS types: `../types/`.
