# app-clip/Sources

SPM source root for the RVF App Clip. Two targets live here, mirroring
the `Package.swift` declarations.

## Subdirectories

- `AppClip/` — Swift target with the SwiftUI App Clip UI (entry, view,
  RVQS seed decoder).
- `RVFBridge/` — C module exposing the RVF FFI header to Swift via a
  module map.

## Build

Targets are built from the parent's `Package.swift`:

```bash
swift build         # macOS syntax check
xcodebuild ...      # full iOS Clip build (see ../Package.swift)
```

## Related

- `../Package.swift` — target/dependency wiring
