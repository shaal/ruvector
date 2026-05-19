# rvf-runtime/examples

Runnable examples (`cargo run --example <name> -p rvf-runtime`).

## Files

- `qr_seed_bootstrap.rs` — full QR Cognitive Seed pipeline: LZ-compress a WASM microkernel → build RVQS payload with HMAC-SHA256 → verify QR fit (≤2,953 bytes) → parse seed, verify signature/hash → decompress microkernel → simulate progressive bootstrap.
- `qr_seed_encode.rs` — focused encoder demo for the QR seed format.
- `capability_report.rs` — print runtime capability flags (features compiled in, hardware probe).
