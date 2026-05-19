# ruvllm/benches

Criterion benchmarks for the LLM runtime. Each file is a separate `[[bench]]`
target. Run with `cargo bench -p ruvllm`.

## Files
- `attention_bench.rs` - attention kernel throughput.
- `rope_bench.rs` - RoPE application speed.
- `matmul_bench.rs` - matmul throughput.
- `norm_bench.rs` - RMSNorm / LayerNorm throughput.
- `moe_bench.rs` - MoE routing + expert dispatch overhead.
- `lora_bench.rs` - LoRA adapter overhead.
- `pi_quant_bench.rs` - Pi-quantization speed.
- `turbo_quant_bench.rs` - alternate / fused quantization path.
- `ane_bench.rs` - Apple Neural Engine path benchmarks.
- `metal_bench.rs` - Metal GPU path benchmarks.
- `serving_bench.rs` - continuous-batching engine throughput.
- `e2e_bench.rs` - end-to-end inference latency.
- `ruvltra_benchmark.rs` - RuvLTRA-Small/Medium full-stack benchmark.
