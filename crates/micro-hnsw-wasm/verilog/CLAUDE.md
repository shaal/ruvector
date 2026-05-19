# micro-hnsw-wasm/verilog

Verilog mirror of the WASM kernel — same multi-core neuromorphic HNSW algorithm in synthesisable RTL for ASIC / FPGA targets.

- `micro_hnsw.v` — single Verilog module matching the Rust constants in `../src/lib.rs` (`MAX_VECTORS`, `MAX_DIMS`, `MAX_NEIGHBORS`, `BEAM_WIDTH`).

Kept side-by-side with the WASM build so the two implementations stay in sync. See parent `../CLAUDE.md`.
