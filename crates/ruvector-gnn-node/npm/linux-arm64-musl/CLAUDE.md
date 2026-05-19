# ruvector-gnn-node/npm/linux-arm64-musl

Per-platform npm sub-package for the NAPI prebuilt binary targeting **linux-arm64-musl**.

## Files
- `package.json` - npm metadata for this platform; declares `os`/`cpu`
  fields so npm picks it up only on matching hosts.
- `ruvector-gnn.linux-arm64-musl.node` - prebuilt NAPI addon (only present after CI build).
