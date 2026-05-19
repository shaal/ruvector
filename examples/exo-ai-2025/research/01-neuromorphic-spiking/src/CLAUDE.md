# 01-neuromorphic-spiking/src

## Files

- `lib.rs` — public re-exports of the spiking primitives.
- `bit_parallel_spikes.rs` — bit-parallel spike kernels (each `u64` =
  64 binary spike states; XOR/AND ops fan out to 64 simulated neurons).
- `spiking_consciousness.rs` — consciousness measurement (Phi-style)
  computed over bit-parallel spike trains.

## Related

- `../benches/spike_benchmark.rs`
- `../examples/quick_bench.rs`
