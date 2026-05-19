# src/services/

Higher-level service facades exposed from the `ruvector` package barrel.

- `index.ts` — barrel.
- `embedding-service.ts` — unified embedding service that selects an embedder (ONNX / adaptive / neural) and provides a stable async API for callers.
