# rvm/crates

The 13 RVM workspace member crates that together implement the coherence-native microhypervisor.

| Crate | Role |
|-------|------|
| `rvm-types` | Foundation types (zero external deps). |
| `rvm-hal` | Hardware abstraction layer; per-arch impls (see `rvm-hal/src/aarch64/`). |
| `rvm-cap` | Capability system with P1/P2 proofs. |
| `rvm-witness` | 64-byte witness records + FNV-1a hash chain. |
| `rvm-proof` | Proof-gated state transitions (Hash / Witness / Zk). |
| `rvm-partition` | Partition object model + split / merge / migrate. |
| `rvm-sched` | Coherence-aware scheduler. |
| `rvm-memory` | Four-tier memory (Hot/Warm/Dormant/Cold). |
| `rvm-coherence` | Coherence + Phi computation. |
| `rvm-boot` | 7-phase deterministic boot. |
| `rvm-wasm` | Optional WebAssembly guest runtime. |
| `rvm-security` | Unified security gate. |
| `rvm-kernel` | Top-level integration; produces the `rvm` binary. |

Dependency direction is strictly layered: every crate eventually rests on `rvm-types`. See each crate's own `CLAUDE.md` and the workspace overview in `../CLAUDE.md`.
