# ruvix-dma/src

## Files

- `lib.rs` — crate root; re-exports the DMA traits/types.
- `controller.rs` — `DmaController` trait defining the controller interface.
- `channel.rs` — `DmaChannel` representing a single DMA channel and its state.
- `buffer.rs` — `DmaBuffer`: cache-coherent memory buffer for transfers.
- `descriptor.rs` — `DmaDescriptor` for scatter-gather linked transfers.
- `config.rs` — `DmaConfig`, `DmaDirection`.
- `error.rs` — DMA error enum.
