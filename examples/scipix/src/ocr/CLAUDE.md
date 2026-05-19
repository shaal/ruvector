# scipix/src/ocr

OCR engine for scipix.

## Files

- `mod.rs` - Module surface.
- `engine.rs` - High-level `OcrEngine` (wraps model + decoder + cache).
- `models.rs` - Model registry / metadata.
- `inference.rs` (~27 KB) - ONNX inference via `ort` (load-dynamic).
- `decoder.rs` - Token/sequence decoder.
- `confidence.rs` - Confidence scoring.

## Related

- Math layer: `../math/`.
- Cache: `../cache/`.
- Bench: `../../benches/ocr_latency.rs`, `../../benches/inference.rs`.
- Models: install via `../../scripts/download_models.sh`.
