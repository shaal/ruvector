# rvf-ebpf/src

Source.

## Files

- `lib.rs` — sole source. `EbpfError` enum (`ClangNotFound`, `CompilationFailed(stderr)`, IO errors), `EbpfCompiler` (clang invocation), `CompiledProgram` (ELF bytes + metadata + SHA3-256 hash). Embeds and exposes the BPF C sources from `../bpf/`.
