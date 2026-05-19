# docs/research/quantization-edge/

Numbered research series on edge-friendly quantization for ruvllm: ultra-low-bit quantization, QAT, QuIP, MoE memory routing, ruvllm quantization architecture, and KV-cache compression.

## Docs

- `00-README.md` - section README.
- `01-ultra-low-bit-quantization-survey.md` - survey of ultra-low-bit quantization.
- `02-quantization-aware-training-qat.md` - QAT.
- `03-quip-2bit-framework.md` - QuIP 2-bit framework.
- `04-moe-memory-aware-routing.md` - MoE memory-aware routing.
- `05-ruvllm-quantization-architecture.md` - ruvllm quantization architecture.
- `06-implementation-plan-rust-ruvllm.md` - Rust implementation plan.
- `07-3int-pi-constant-quantization.md` - 3-int Pi-constant quantization.
- `08-turboquant-kv-cache-compression.md` - TurboQuant KV cache compression.
- `09-triattention-kv-sparsity.md` - TriAttention KV sparsity.
- `10-stacked-kv-compression.md` - stacked KV compression.

## Related

- `../../ruvllm/` - canonical ruvllm docs.
- `../../adr/ADR-147-stacked-kv-cache-triattention-turboquant.md`, `ADR-154-rabitq-rotation-binary-quantization.md`.
- `../../adr/` ADR-181 - pi quant BitNet integration.
