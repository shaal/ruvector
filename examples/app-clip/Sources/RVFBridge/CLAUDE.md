# app-clip/Sources/RVFBridge

C module that bridges the RVF Rust FFI to Swift. Pure interface shim —
no implementation lives here; the implementation is the
`librvf_runtime.a` static lib produced by `cargo build --target
aarch64-apple-ios`.

## Files

- `module.modulemap` — declares the `RVFBridge` Clang module and
  publishes `rvf_bridge.h` as its umbrella header so Swift can
  `import RVFBridge`.
- `rvf_bridge.h` — C declarations for the RVF runtime FFI surface used
  by the App Clip (seed parse, decode, error reporting).

## Build

Built indirectly by the parent SPM target — see `../../Package.swift`.
Linker flags in that manifest point to
`../../target/aarch64-apple-ios/release` for the `.a`.

## Related

- `../AppClip/SeedDecoder.swift` — primary consumer of these symbols
- `../../Package.swift` — target / linker config
