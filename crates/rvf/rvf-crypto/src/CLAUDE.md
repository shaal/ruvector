# rvf-crypto/src

Source.

## Files

- `lib.rs` — `no_std` shim, module decls, public re-exports.
- `hash.rs` — SHAKE-256 hashing (`shake256_128`, `shake256_256`, `shake256_hash`).
- `sign.rs` — Ed25519 signing/verification (feature-gated `ed25519`).
- `footer.rs` — encode/decode signature footer block appended to RVF files.
- `witness.rs` — WITNESS_SEG chain construction/verification (SHAKE-256-linked).
- `attestation.rs` — TEE attestation: `QuoteVerifier`, `TeeBoundKeyRecord`, header/record codecs, witness payload build/verify, key-binding verification.
- `lineage.rs` — lineage/derivation records and verification.
