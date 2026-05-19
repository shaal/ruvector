# ruvector-gnn-node/npm/win32-x64-msvc

Per-platform npm sub-package for the NAPI prebuilt binary targeting **win32-x64-msvc**.

## Files
- `package.json` - npm metadata for this platform; declares `os`/`cpu`
  fields so npm picks it up only on matching hosts.
- `ruvector-gnn.win32-x64-msvc.node` - prebuilt NAPI addon (only present after CI build).
