Cross-platform native binary build scripts, primarily for the NAPI-RS bindings published under `../../npm/`.

Files:
- `build-all-platforms.sh` - builds NAPI-RS bindings for all supported target triples.
- `build-linux.sh` - Linux-only NAPI build.
- `copy-binaries.sh` - copies built native binaries into the npm package layout for publishing.

Run from the repo root. Companion publish scripts live in `../publish/`.
