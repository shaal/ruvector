# hailort-sys

Raw FFI bindings for Hailo's HailoRT C library (ADR-167) — links `libhailort` for Pi 5 + AI HAT+ NPU acceleration. Without the `hailo` feature, the crate is an empty stub so `cargo check` succeeds on x86 hosts without HailoRT installed.

## Important files

- `Cargo.toml` — `links = "hailort"` to prevent duplicate linkage. `[build-dependencies] bindgen = "0.71"`. Feature `hailo` triggers real bindgen generation; default is stub.
- `build.rs` — When `hailo` feature is set, runs `bindgen` against `wrapper.h` to generate `bindings.rs` in `OUT_DIR`. Also emits the `cargo:rustc-link-lib=hailort` directive.
- `wrapper.h` — Single-header include that pulls `<hailo/hailort.h>`; the input to bindgen.
- `src/lib.rs` — Re-`include!`s the generated bindings and provides `version_triple()` smoke-test (calls `hailo_get_library_version`).
- `Cargo.lock` — Lockfile (crate uses a standalone-like setup; lock present).

## Public API

- `version_triple() -> Option<(u32, u32, u32)>` — returns library version (major, minor, revision) or `None` on error; returns `(0,0,0)` when `hailo` feature disabled.
- Generated FFI: `hailo_*` types/functions from the C header (when feature enabled).

## Build / Feature notes

- Default build: stub only, no system deps.
- `--features hailo`: requires system `libhailort.so` and `<hailo/hailort.h>` headers.
- ADR-167 §5 step 3, branch `hailo-backend`. Iter 219 rejoined parent workspace (Gap E folded into Gap B).

## Related

- Consumed by edge-AI/inference crates targeting the Hailo-8 NPU on Raspberry Pi 5.
- Companion of `ruos-thermal` for Pi 5 supervision.
