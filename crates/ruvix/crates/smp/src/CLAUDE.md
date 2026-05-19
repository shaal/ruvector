# ruvix-smp/src

## Files

- `lib.rs` — crate root; re-exports the SMP primitives.
- `cpu.rs` — `CpuId` newtype + `CpuState` lifecycle state machine.
- `topology.rs` — `CpuTopology` for system-wide CPU state tracking.
- `percpu.rs` — `PerCpu<T>` per-CPU storage indexed by `CpuId`.
- `spinlock.rs` — `SpinLock<T>`: ticket-based, fair spinlock.
- `ipi.rs` — `IpiMessage` and inter-processor-interrupt send/recv helpers.
- `barriers.rs` — ARM64 memory barriers: `dmb()`, `dsb()`, `isb()`, `sev()`, `wfe()`.
