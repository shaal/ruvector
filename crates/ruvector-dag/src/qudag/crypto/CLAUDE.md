# ruvector-dag/src/qudag/crypto

Quantum-resistant cryptography for QuDAG. Enable `feature = "production-crypto"` on `ruvector-dag` to switch placeholders (HMAC-SHA256 / HKDF-SHA256) for production ML-DSA / ML-KEM.

| Component | With `production-crypto` | Without |
|-----------|--------------------------|---------|
| ML-DSA-65 | Dilithium3 | HMAC-SHA256 placeholder |
| ML-KEM-768 | Kyber768 | HKDF-SHA256 placeholder |
| Differential Privacy | Production | Production |
| Keystore | `zeroize` | `zeroize` |

## Files

- `mod.rs` — re-exports; call `check_crypto_security()` at startup to log security posture.
- `ml_dsa.rs` — ML-DSA-65 signing wrappers.
- `ml_kem.rs` — ML-KEM-768 key encapsulation wrappers.
- `identity.rs` — `QuDagIdentity`, `IdentityError`.
- `keystore.rs` — zeroising key storage.
- `differential_privacy.rs` — `DifferentialPrivacy`, `DpConfig`.
- `security_notice.rs` — `check_crypto_security()` runtime notice.

See `../CLAUDE.md`.
