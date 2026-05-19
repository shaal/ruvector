# rvf-kernel/src

Source.

## Files

- `lib.rs` — top-level docs + module decls; sketches `KernelBuilder` / `KernelVerifier` flow.
- `config.rs` — `KernelConfig` knobs (arch, source path, options).
- `docker.rs` — reproducible kernel compilation in a Docker container.
- `initramfs.rs` — build a gzipped cpio newc archive with a real `/init` script.
- `error.rs` — `KernelError`.
