# ruvix-queue/src

## Files

- `lib.rs` — crate root, queue trait + manager re-exports.
- `ring.rs` — base SPSC/MPMC ring buffer.
- `ring_optimized.rs` — cache-line padded variant tuned for hot SQ/CQ paths.
- `descriptor.rs` — zero-copy descriptor (offset+length) types with TOCTOU policy checks.
- `kernel_queue.rs` — kernel-facing `KernelQueue` integrating ring + descriptors + capabilities.
