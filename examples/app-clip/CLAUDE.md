# app-clip

iOS App Clip skeleton that consumes the RVF (Ruvector File) runtime via
its C FFI. Designed to scan QR/NFC RVQS cognitive-seed codes and decode
them on-device while staying under Apple's 15 MB App Clip size limit.
This is a Swift Package + bridging-header scaffold, not a runnable demo
on its own — you build the Rust static lib first.

## Important files

- `Package.swift` — SPM manifest declaring two targets:
  - `RVFBridge` — C module that exposes `Sources/RVFBridge/rvf_bridge.h`
    to Swift and links the pre-built static lib
    (`librvf_runtime.a`).
  - `AppClip` — SwiftUI App Clip target depending on `RVFBridge`.
  Linker flags assume the Rust artifact lives under
  `../../target/aarch64-apple-ios/release`.

## Build

```bash
# 1. Build the Rust static library for iOS device.
cargo build --release --target aarch64-apple-ios --lib

# 2. Open Package.swift in Xcode and build the AppClip target,
#    or use `swift build` for syntax checking (won't link the iOS lib
#    on non-Mac hosts).
```

## Tech stack

- Swift 5.9, iOS 16+, SwiftUI
- Rust → C FFI via `librvf_runtime.a` (built by another workspace crate)
- C module map (`module.modulemap`) bridges the FFI header to Swift

## Related

- `Sources/AppClip/` — SwiftUI entry, view, and seed decoder
- `Sources/RVFBridge/` — header and modulemap for the Rust FFI
- The Rust library producing `librvf_runtime.a` lives elsewhere in the
  monorepo (search for `rvf_runtime` cdylib/staticlib in `crates/`).
