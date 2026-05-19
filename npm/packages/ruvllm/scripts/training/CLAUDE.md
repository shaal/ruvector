# scripts/training/

Training data + generators for the `@ruvector/ruvllm` self-learning router/embedding models.

## Capability JSON datasets

- `agentic-flow-capabilities.json`, `claude-flow-capabilities.json`, `ruvector-capabilities.json` — capability manifests for each ecosystem.

## Generators

- `claude-code-synth.js` — synthesizes Claude-code-style training samples.
- `claude-hard-negatives.js` — generates hard negative pairs.
- `contrastive-finetune.js` — contrastive fine-tuning driver.
- `generate-ecosystem-triplets.js` — produces the `ecosystem-triplets.jsonl` corpus (~170 KB).
- `routing-dataset.js` — assembles the router-training dataset.

## Validation

- `validate-ecosystem.js` — validates the ecosystem triplets.
- `validation-results.json` — last validation run output.

## Generated data

- `ecosystem-triplets.jsonl` — anchor/positive/negative triplets for contrastive training.
