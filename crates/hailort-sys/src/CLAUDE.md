# hailort-sys/src

Only one Rust source file; everything else (the actual FFI bindings) is generated at build time into `OUT_DIR`.

## Files

- `lib.rs` — `#![allow(non_upper_case_globals, non_camel_case_types, non_snake_case, dead_code)]` to accommodate bindgen output. `include!(concat!(env!("OUT_DIR"), "/bindings.rs"));` pulls in the generated FFI. Exposes `version_triple()` as a smoke-test wrapper around `hailo_get_library_version`.

## Notes

- Without `--features hailo`, `OUT_DIR/bindings.rs` is a stub written by `build.rs`, and `version_triple` returns `Some((0,0,0))`.
- Never edit generated bindings; modify `wrapper.h` or `build.rs` instead.
