Python-based comparative benchmark for ruvector's quantization stack against standard ANN datasets (GloVe, SIFT, etc.). Independent of the TS load-testing suite in `../src/`.

Files:
- `benchmark_quantized_search.py` - measures recall@k, compression ratio, and latency across all quantization tiers in `ruvector-core` (ScalarQuantized, Int4, Product, Binary) and `ruvllm`'s TurboQuant. Runs each config 3x with independent seeds to report variance.
- `ANALYSIS.md` - write-up of findings. Key result: the two quantization subsystems (`ruvllm/turbo_quant` and `ruvector-core/quantization`) are disconnected from HNSW search; `QuantizedVector::distance` is never called during graph traversal, so all quantization is storage-only at present.
- `results/` - JSON + markdown output of the latest run.
