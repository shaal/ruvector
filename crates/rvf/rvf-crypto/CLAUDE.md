# rvf-crypto

Cryptographic primitives for the RuVector Format: SHAKE-256 hashing, optional Ed25519 segment signing/verification, signature footer codec, and WITNESS_SEG audit-trail support. `no_std` compatible (alloc required); Ed25519 gated behind the default `ed25519` feature.

## Layout

- `Cargo.toml` — features: `default = ["std", "ed25519"]`, `std`, `ed25519`. Deps: `rvf-types`, `sha3` (no_std-compatible), optional `ed25519-dalek`.
- `src/lib.rs` — `no_std` shim, module decls, public re-exports.
- `src/hash.rs` — `shake256_128`, `shake256_256`, `shake256_hash`.
- `src/sign.rs` — Ed25519 sign / verify (feature-gated).
- `src/footer.rs` — `encode_signature_footer` / `decode_signature_footer`.
- `src/witness.rs` — SHAKE-256-linked WITNESS_SEG chains.
- `src/attestation.rs` — TEE attestation records, `QuoteVerifier`, `TeeBoundKeyRecord`, witness payload build/verify.
- `src/lineage.rs` — derivation/lineage records linking parent ↔ child stores.

## Public API

`shake256_*`, `encode_signature_footer`/`decode_signature_footer`, attestation primitives (`QuoteVerifier`, `TeeBoundKeyRecord`, `VerifiedAttestationEntry`, `attestation_witness_entry`, `encode_*`/`decode_*` headers and records, `verify_key_binding`), plus witness/lineage helpers.

## Related

- `../rvf-types` — base types
- `../rvf-runtime`, `../rvf-cli`, every `rvf-adapter-*` use this for signing / audit
