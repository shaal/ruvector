# rvf-ebpf

Real eBPF programs for RVF vector distance computation and traffic control. Provides BPF C sources, an `EbpfCompiler` shim that invokes `clang` to produce ELF objects, and a `CompiledProgram` wrapper with SHA3-256 integrity hash for embedding into RVF stores (`EBPF_SEG`).

## Layout

- `Cargo.toml` — deps: `rvf-types`, `sha3`, `tempfile`.
- `src/lib.rs` — sole Rust source. Defines `EbpfError`, `EbpfCompiler`, `CompiledProgram`. Loads/compiles the BPF C sources in `bpf/`.
- `bpf/` — eBPF C sources (compiled via `clang`).

## bpf/

- `xdp_distance.c` — XDP program computing vector distance at the NIC.
- `socket_filter.c` — socket-level port filter.
- `tc_query_route.c` — TC-layer query priority routing.
- `vmlinux.h` — kernel type definitions (CO-RE).

## Public API

`EbpfError`, `EbpfCompiler`, `CompiledProgram`, and the `programs` module's accessors for the bundled C sources. Uses `rvf_types::ebpf::{EbpfAttachType, EbpfHeader, EbpfProgramType, EBPF_MAGIC}` for segment layout.

## Related

- `../rvf-types::ebpf` — header types
- `../rvf-cli` `embed-ebpf` subcommand consumes this
