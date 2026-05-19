# examples/google-cloud

RuVector Cloud Run GPU benchmark suite (`ruvector-cloudrun-gpu`). Builds GPU/CPU benchmarks for vector operations and self-learning models, packaged as a Cloud Run service with GPU + Raft cluster support.

## Key files
- `Cargo.toml` - Binary crate `ruvector-cloudrun-gpu` (single `gpu-benchmark` bin). Depends on `ruvector-core`, `ruvector-gnn`, `ruvector-attention`, `ruvector-graph` (wasm feature); axum/tower for the server, `clap` CLI, `hdrhistogram`, `tokio`, `rayon`.
- `Dockerfile.build` - Multi-stage build image.
- `Dockerfile.cloudrun` - Slim runtime image for Cloud Run.
- `Dockerfile.gpu` - GPU-enabled runtime image (CUDA).
- `Dockerfile.simple` - Minimal CPU-only image.
- `cloudrun.yaml` - Cloud Run service manifest.
- `deploy.sh` - GCP deploy script (env-driven: project, region, memory, GPU type/count, Raft cluster size, etc.).
- `src/` - Rust source (see its CLAUDE.md).
- `benchmark_results/` - Saved benchmark JSON outputs.

## Run
- Local CLI: `cargo run --release -- quick --dims 128 --num-vectors 10000`.
- Container build: `docker build -f Dockerfile.gpu -t ruvector-bench .`.
- GCP deploy: `./deploy.sh` (override env vars like `GCP_PROJECT_ID`, `GPU_TYPE`, `CLUSTER_SIZE`).

## Tech stack
- Rust 2021, Tokio, Axum, Clap, hdrhistogram, rayon.
- CUDA (optional), Google Cloud Run with NVIDIA L4 GPUs (default).

## Related
- `crates/ruvector-core`, `crates/ruvector-gnn`, `crates/ruvector-attention`, `crates/ruvector-graph`.
- For non-cloud benchmarks see `examples/vibecast-7sense/crates/sevensense-benches/`.
