# ruvector-attention/src/info_bottleneck

Information-bottleneck attention: penalise mutual information between query and key beyond what's necessary for the output.

## Files

- `mod.rs` — module entry.
- `bottleneck.rs` — IB-regularised attention.
- `kl_divergence.rs` — KL helpers used by the regulariser.
