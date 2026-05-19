# ruvix-smp

Symmetric Multi-Processing primitives for the RuVix Cognition Kernel (ADR-087 Phase C). Supports up to 256 CPUs with efficient
per-CPU data structures and synchronization. Provides ARM64 memory barriers (DMB / DSB / ISB / SEV / WFE).

## Core components

| Component | Purpose |
|---|---|
| `CpuId` | Newtype for CPU identifiers (0-255) |
| `CpuState` | State machine for CPU lifecycle |
| `PerCpu<T>` | Per-CPU data storage indexed by CPU ID |
| `CpuTopology` | System-wide CPU state tracking |
| `SpinLock<T>` | Ticket-based spinlock with fairness guarantees |
| `IpiMessage` | Inter-processor interrupt message types |

## Files

- `Cargo.toml` — depends on `ruvix-types` + `ruvix-hal`. Dev: proptest.
- `README.md` — public docs.
- `src/` — see `src/CLAUDE.md`.
