# ruvix-drivers/src

## Files

- `lib.rs` — crate root; re-exports the per-device modules.
- `pl011.rs` — `Pl011` UART driver implementing `ruvix_hal::Console`.
- `gic.rs` — `Gic` (GICv2 / GIC-400) implementing `ruvix_hal::InterruptController`.
- `timer.rs` — `ArmGenericTimer` implementing `ruvix_hal::Timer`.
- `mmio.rs` — shared MMIO read/write helpers used by the drivers above.
