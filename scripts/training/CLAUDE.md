Training pipeline scripts for the RuvLTRA LoRA fine-tuning workflow and the JS deobfuscator model. Mixture of Python (training/eval) and Node `.mjs` (data generation).

Key Python entry points:
- `run_calibration.py` - Phase 1: quantization calibration + TurboQuant profiling. Downloads a HF model, generates code-focused calibration data, produces quantized GGUF variants.
- `run_sft.py` - Phase 2: LoRA SFT fine-tuning with `peft` + `transformers`, merges adapter, converts to GGUF, runs the release gate.
- `release_gate.py` - implements the 7 ship/no-ship criteria from ADR-129 Section 3.2.
- `evaluate-model.py` - model evaluation.
- `contamination_check.py` - eval/train contamination detector.
- `build-optimal-dataset.py`, `build-optimal-dataset-v2.py` - balance learnability with diversity when constructing the corpus.
- `filter-and-augment.py` - corpus filtering + augmentation.
- `export_training_data.py`, `export-to-rvf.py`, `export-weights-bin.py` - export trained artifacts (training JSONL, RVF container, raw weights).
- `train-deobfuscator.py`, `train-deobfuscator-v2.py` - JS deobfuscator model training.

Node generators:
- `generate-data-v2.mjs`, `generate-deobfuscation-data.mjs`, `extract-sourcemap-pairs.mjs` - synthesize deobfuscation training pairs.

Shell orchestrators:
- `launch-gpu-training.sh` - kicks off GPU training jobs.
- `deploy_training.sh` - deploys the training container.
- `nightly_train.sh` - cron-style nightly retraining.

Containers:
- `Dockerfile` - RuvLTRA pipeline image (CUDA 12.4 + Python 3.11 + GGUF + llama-cpp-python). Targets Cloud Run Jobs with L4 GPUs.
- `Dockerfile.deobfuscator` - PyTorch 2.2 + CUDA 12.1 image for deobfuscator training/export.

Subdirectory:
- `data/training/` - bundled `merged_corpus.jsonl` (duplicate of `../../data/training/merged_corpus.jsonl`, included so the Dockerfile build context contains it).

Related: `../../data/training/` (input corpora), `../../scripts/training_orchestrator.sh`, ADR-129.
