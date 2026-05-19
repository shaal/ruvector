Integration tests that pull the *published* `ruvector-attention` Rust crate, NAPI package, and WASM package from registries and verify they actually work end-to-end inside a fresh Docker container.

Files:
- `Cargo.toml` - standalone bin crate `ruvector-attention-integration-test` depending on `ruvector-attention = "0.1.0"` from crates.io (not a workspace path dep).
- `src/main.rs` - Rust runner exercising ScaledDotProduct, MultiHead, Hyperbolic, Linear, Flash, LocalGlobal, MoE, and Graph attention.
- `package.json` - npm bundle pulling `ruvector-attention-wasm@0.1.0` and `@ruvector/attention@0.1.0`. Scripts: `test` (node --test), `test:wasm`, `test:napi`.
- `test-napi.mjs` - NAPI binding smoke test.
- `test-wasm.mjs` - WASM binding smoke test.
- `test_sparql_pr66.sql` - SPARQL fixture used by the PR66 regression scenario.
- `Dockerfile` - container that installs the published packages and runs all three test entry points.

Documentation / reports:
- `FINAL_REVIEW_REPORT.md`, `FINAL_SUMMARY.md`, `FIXES_APPLIED.md`, `PR66_REVIEW_COMMENT.md`, `PR66_TEST_REPORT.md`, `PUBLICATION_COMPLETE.md`, `ROOT_CAUSE_AND_FIX.md`, `SUCCESS_REPORT.md`, `ZERO_WARNINGS_ACHIEVED.md` - run/review notes from PR66.

Run locally: `docker build -t ruvector-attention-it . && docker run ruvector-attention-it`. Or run each entry point directly after publishing.
