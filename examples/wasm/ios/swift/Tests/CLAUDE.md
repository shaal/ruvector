# wasm/ios/swift/Tests

XCTest suite for the Swift consumer of `ruvector-ios-wasm`.

## Files

- `RecommendationTests.swift` - Tests covering the recommendation engine flow against `Resources/recommendation.wasm`.

## How to run

```bash
cd /home/user/ruvector/examples/wasm/ios/swift
swift test
```

## Related

- Code under test: `../WasmRecommendationEngine.swift`, `../HybridRecommendationService.swift`.
