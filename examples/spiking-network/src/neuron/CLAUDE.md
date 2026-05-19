# spiking-network / src / neuron

Neuron models behind a common trait.

## Important files
- `mod.rs` - module root and re-exports.
- `traits.rs` - the `Neuron` trait (state update, spike emission, reset).
- `lif.rs` - Leaky Integrate-and-Fire neuron (canonical simple model).
- `izhikevich.rs` - Izhikevich neuron (rich spiking dynamics with low compute).

## Related
- Used by `../network/`. Inputs come from `../encoding/`.
