# ruvix-bcm2711/src

## Files

- `lib.rs` — crate root; SoC memory-map constants + driver re-exports.
- `mmio.rs` — base MMIO helpers / addresses for the SoC.
- `gpio.rs` — GPIO driver (pin mux, level, IRQ).
- `mini_uart.rs` — Mini UART (PL011 alternative used during early boot).
- `mailbox.rs` — VideoCore mailbox interface for firmware property requests.
- `interrupt.rs` — SoC-specific interrupt controller bits.
