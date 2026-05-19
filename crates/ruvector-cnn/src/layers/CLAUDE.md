# ruvector-cnn/src/layers

Full FP and quantized neural-network layers used by the backbone and standalone embedders.

## Files

- `mod.rs` — module entry + layer re-exports.
- `conv.rs` — FP conv2d.
- `linear.rs` — FP linear / fully connected.
- `batchnorm.rs` — batch normalization.
- `activation.rs` — activations (ReLU, ReLU6, Hard-Swish, Sigmoid, etc.).
- `pooling.rs` — max / avg pooling.
- `quantized_conv2d.rs` — INT8 conv2d using `src/kernels/`.
- `quantized_depthwise.rs` — INT8 depthwise conv2d.
- `quantized_linear.rs` — INT8 linear.
- `quantized_pooling.rs` — INT8 pooling.
- `quantized_residual.rs` — INT8 residual / shortcut.
