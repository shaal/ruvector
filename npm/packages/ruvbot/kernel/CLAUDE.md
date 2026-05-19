# ruvbot / kernel

Pre-built Linux kernel image consumed by the RVF (Rust Virtual
Function) sandbox runtime that hosts `ruvbot.rvf`.

## Files
- `bzImage` - Bootable Linux kernel binary. Loaded by the rvf runtime
  via `scripts/run-rvf.js` when launching the bundled function image.

Only `kernel/bzImage` is included in the published npm tarball (see
`files` in `../package.json`).
