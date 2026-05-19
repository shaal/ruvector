# ruvector-cnn/src/backbone

MobileNet-V3 backbone (Small + Large). Behind the `backbone` feature.

## Files

- `mod.rs` — backbone module entry; selects V3-Small vs V3-Large.
- `mobilenet.rs` — MobileNet-V3 model construction.
- `blocks.rs` — inverted residual + SE blocks used by MobileNet-V3.
- `layer.rs` — generic typed-layer wrapper used by the backbone graph.
