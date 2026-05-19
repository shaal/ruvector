# ruvector-fpga-transformer/src/artifact

Signed model-artifact format. Artifacts contain a manifest with shape and quantization metadata plus weight blobs, signed via ed25519.

## Files

- `mod.rs` — module entry + `ModelArtifact` public type.
- `manifest.rs` — manifest schema (shape, quantization, hashes).
- `pack.rs` — packer (writes a signed artifact).
- `verify.rs` — verifier (checks signature + sha256 hashes before load).
