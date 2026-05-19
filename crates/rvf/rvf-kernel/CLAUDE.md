# rvf-kernel

Real Linux microkernel builder for RVF cognitive containers. Builds, verifies, and embeds Linux kernel images (bzImage / ELF) inside RVF files via the KERNEL_SEG segment. Supports prebuilt loads, reproducible Docker builds, valid newc cpio initramfs construction, and SHA3-256 integrity verification.

## Layout

- `Cargo.toml` — deps: `rvf-types` (`std`), `sha3`, `flate2`. Dev: `tempfile`.
- `src/lib.rs` — module decls + top-level docs sketching the `KernelBuilder` / `KernelVerifier` pipeline.
- `src/config.rs` — `KernelConfig` (arch, source, options).
- `src/docker.rs` — reproducible Docker-driven kernel build.
- `src/initramfs.rs` — gzipped cpio (newc) initramfs builder including a real `/init` script.
- `src/error.rs` — `KernelError` enum.

## Public API

`KernelBuilder` (`from_prebuilt`, `build_docker`, `build_initramfs`, `embed`), `KernelVerifier::verify`, `KernelConfig`, `KernelError`.

## Related

- `../rvf-launch` — QEMU launcher that extracts and boots these kernels
- `../rvf-types::kernel` — KERNEL_SEG header types
- `../rvf-cli` `embed-kernel` and `launch` subcommands
