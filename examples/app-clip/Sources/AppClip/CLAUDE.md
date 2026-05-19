# app-clip/Sources/AppClip

SwiftUI App Clip target. Handles invocation (QR scan / App Clip Code /
NFC), decodes an RVQS cognitive seed through the RVF C bridge, and
renders a minimal status view. Skeleton code — wire-frames the flow but
depends on the pre-built `librvf_runtime.a`.

## Files

- `AppClipApp.swift` — `@main` SwiftUI `App`; owns an `AppClipState`
  `@StateObject` and handles the `https://rvf.example.com/seed?id=...`
  invocation URL scheme.
- `AppClipView.swift` — primary view; QR camera flow, status display,
  and result rendering (~11 KB, the bulk of the UI).
- `SeedDecoder.swift` — Swift wrapper that calls into the FFI symbols
  declared in `../RVFBridge/rvf_bridge.h` to decode an RVQS payload.

## Tech stack

- SwiftUI, AVFoundation (camera), CoreNFC (tag reads)
- Depends on the `RVFBridge` C target (sibling)

## Related

- `../RVFBridge/` — C header / module map this target imports
- `../../Package.swift` — target declaration
