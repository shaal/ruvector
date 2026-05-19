# rvf-launch/src

Source.

## Files

- `lib.rs` — `LaunchConfig` (path to RVF store, QEMU options) + top-level launcher.
- `extract.rs` — read KERNEL_SEG out of an RVF file into a temp bzImage.
- `qemu.rs` — build the QEMU `Command`, set up stdio, spawn the `Child`.
- `qmp.rs` — QMP JSON protocol client over `TcpStream` for `query-status`, `system_powerdown`, `quit`.
- `error.rs` — `LaunchError`.
