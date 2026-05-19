# scipix/examples

Runnable example programs for scipix plus a sample dataset and a browser WASM demo.

## Files

- `simple_ocr.rs` - Smallest OCR call.
- `batch_processing.rs` - Batch over many images.
- `streaming.rs` - Streaming pipeline.
- `custom_pipeline.rs` - Building a custom preprocess/OCR/post-process pipeline.
- `api_server.rs` - Embedded HTTP API server example.
- `accuracy_test.rs` - Accuracy evaluation harness.
- `lean_agentic.rs` - Lean agentic loop combining OCR with reasoning.
- `optimization_demo.rs` - SIMD/quantization/parallel demo.
- `sample_dataset.json` - Small fixture dataset.
- `wasm_demo.html` - Browser demo loading the WASM build.

## How to run

```bash
cargo run -p ruvector-scipix --example simple_ocr
cargo run -p ruvector-scipix --example api_server
# WASM demo:
cd /home/user/ruvector/examples/scipix/web && ./build.sh && python3 -m http.server 8080
```

## Related

- Bins: `../src/bin/`.
- WASM build instructions: `../BUILD_WASM.md`, `../docs/WASM_QUICK_START.md`.
