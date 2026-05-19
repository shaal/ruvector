# rvf-ebpf/bpf

eBPF C sources compiled by `EbpfCompiler` in `../src/lib.rs` via `clang`.

## Files

- `xdp_distance.c` — XDP program performing vector-distance computation in the NIC fast-path.
- `socket_filter.c` — socket-level port filter (drop / pass per port).
- `tc_query_route.c` — traffic-control program routing queries by priority class.
- `vmlinux.h` — kernel type definitions for CO-RE BPF builds (auto-generated; do not edit manually).
