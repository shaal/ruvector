Rust binary that imports the published `ruvector-attention` crate (from crates.io) and runs smoke tests for every attention mechanism.

Files:
- `main.rs` - calls `test_scaled_dot_product_attention`, `test_multi_head_attention`, `test_hyperbolic_attention`, `test_linear_attention`, `test_flash_attention`, `test_local_global_attention`, `test_moe_attention`, `test_graph_attention`.

Built and invoked as `test-attention` via the `[[bin]]` entry in `../Cargo.toml`.
