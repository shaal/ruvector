# scipix/src/preprocess

Image preprocessing pipeline before OCR.

## Files

- `mod.rs` - Module surface and pipeline wiring.
- `pipeline.rs` - Composable preprocessing pipeline.
- `deskew.rs` - Deskewing.
- `rotation.rs` - Rotation correction.
- `enhancement.rs` - Contrast / denoise / sharpening.
- `segmentation.rs` - Page / region segmentation.
- `transforms.rs` - Geometric transforms.

## Related

- Bench: `../../benches/preprocessing.rs`.
- Docs: `../../docs/07_IMAGE_PREPROCESSING.md`, `../../docs/PREPROCESSING_API.md`, `../../docs/PREPROCESSING_MODULE.md`.
