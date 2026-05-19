# ruvector-wasm/src/kernel

Kernel pack system (ADR-005): secure, sandboxed execution of ML compute
kernels with manifest validation, signature/hash verification, an
allowlist, and epoch-based execution budgets. Compiled only when the
`kernel-pack` cargo feature is enabled.

## Files
- `mod.rs` - module wiring + public re-exports.
- `manifest.rs` - kernel pack manifest parsing/validation.
- `signature.rs` - Ed25519 signature verification for manifests.
- `hash.rs` - SHA256 hash verification for bundled kernel binaries.
- `allowlist.rs` - trusted-kernel allowlist enforcement.
- `epoch.rs` - epoch-based execution budgets (gas-like accounting).
- `memory.rs` - shared-memory protocol for tensor data crossing the
  host/guest boundary.
- `runtime.rs` - the actual kernel runtime: load -> verify -> execute.
- `error.rs` - kernel-subsystem errors.
