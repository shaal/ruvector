# rvf-launch

QEMU microVM launcher for RVF cognitive containers. Extracts the kernel image from the RVF file's KERNEL_SEG, assembles a QEMU command line, launches the VM, and exposes a handle for query/shutdown/kill via QMP.

## Layout

- `Cargo.toml` — deps: `rvf-types` (`std`), `rvf-runtime`, `serde`/`serde_json`, `tempfile`.
- `src/lib.rs` — `LaunchConfig`, top-level launcher orchestration. Uses `rvf_types::kernel::KernelArch`.
- `src/extract.rs` — KERNEL_SEG extraction to a temp file.
- `src/qemu.rs` — QEMU command-line construction + child-process spawning.
- `src/qmp.rs` — QMP protocol client over `TcpStream` for VM control.
- `src/error.rs` — `LaunchError`.

## Public API

`LaunchConfig`, the launcher entry returning a managed `Child` handle, `LaunchError`.

## Related

- `../rvf-kernel` — produces the KERNEL_SEG this crate extracts
- `../rvf-cli` `launch` subcommand
- `../rvf-runtime` — store API used to open the RVF file
