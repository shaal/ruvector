# ruvix-hal/src

## Files

- `lib.rs` — crate root; re-exports the trait modules below.
- `console.rs` — `Console` trait (serial I/O).
- `interrupt.rs` — `InterruptController` trait (IRQ/FIQ).
- `timer.rs` — `Timer` trait (monotonic time + deadlines).
- `mmu.rs` — `Mmu` trait (page tables + virtual memory).
- `power.rs` — `PowerManagement` trait (CPU power states, reset).
