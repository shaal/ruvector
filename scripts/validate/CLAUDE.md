Pre-publish and pre-deploy validation scripts. Run these before invoking anything in `../publish/` or `../deploy/`.

Files:
- `validate-packages.sh` - full pre-publish validation of all ruvector packages (versions, manifests, build, smoke tests).
- `validate-packages-simple.sh` - quick subset of the same checks for fast feedback.
- `verify-paper-impl.sh` - verifies that paper-referenced implementations build and run as expected.
- `verify_hnsw_build.sh` - confirms the vendored `hnsw_rs` patch (`../../patches/hnsw_rs/`) builds cleanly with the workspace.
