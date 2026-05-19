# ruvix-boot/src

## Files

- `lib.rs` — crate root; orchestrates the 5-stage boot sequence.
- `boot_loader.rs` — top-level `BootLoader` driving the stages.
- `stages.rs` — individual stage implementations (Hardware Init / RVF Verify / Object Create / Component Mount / First Attestation).
- `manifest.rs` — RVF manifest parsing + validation.
- `signature.rs` — ML-DSA-65 (NIST FIPS 204) signature verification. PANIC-on-failure path.
- `mount.rs` — Component Mount stage logic.
- `capability_distribution.rs` — distributes initial capabilities to mounted components, then drops the root task to its minimum
  capability set.
- `attestation.rs` — first-attestation emission.
- `witness_log.rs` — append-only, cryptographically linked witness log.
