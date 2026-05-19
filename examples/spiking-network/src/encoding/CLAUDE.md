# spiking-network / src / encoding

Input encoders that turn analog signals into spike trains for the network.

## Important files
- `mod.rs` - module root; defines the encoding trait and the concrete encoders (rate / temporal / population coded) used by the example binaries.

## Related
- Consumed by `../network/`. Neuron models that receive the encoded spikes: `../neuron/`.
