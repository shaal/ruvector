# ruvix-rpi-boot/src

## Files

- `lib.rs` — crate root; ties early-boot pieces together.
- `early_uart.rs` — minimal UART available before drivers init (uses BCM2711 mini-UART).
- `dtb.rs` — accepts the DTB pointer handed off in `x0` by the GPU firmware and forwards it to `ruvix-dtb`.
- `spin_table.rs` — multi-core spin-table secondary-CPU bring-up.
- `config.rs` — `config.txt` / cmdline-related boot configuration.
