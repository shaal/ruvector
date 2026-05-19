# scripts/

Comparison harnesses, HuggingFace publishing, and training data generation for `@ruvector/ruvllm`.

Top-level model-comparison runners:

- `ensemble-model-compare.js` — ensembled multi-model comparison.
- `hybrid-model-compare.js` — hybrid comparison strategy.
- `improved-model-compare.js`, `optimized-model-compare.js`, `real-model-compare.js` — alternative comparison variants and a "real" (non-mock) comparison.

Subdirectories:

- `huggingface/publish.sh` — script to publish models to HuggingFace.
- `training/` — JSON capabilities, JS generators, validation scripts, and a 170 KB JSONL of ecosystem triplets used to train the routing/embedding models.
