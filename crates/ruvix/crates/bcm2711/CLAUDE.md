# ruvix-bcm2711

Low-level drivers for the Broadcom BCM2711 (Raspberry Pi 4 / Pi 400 / CM4) and BCM2712 (Raspberry Pi 5) SoCs.

## Supported boards

- Raspberry Pi 4 Model B (BCM2711, Cortex-A72)
- Raspberry Pi 5 (BCM2712, Cortex-A76)
- Raspberry Pi 400 (BCM2711)
- Raspberry Pi Compute Module 4 (BCM2711)

## Memory map (RPi 4 / BCM2711)

| Bus address | ARM physical | Description |
|---|---|---|
| 0x7E00_0000 | 0xFE00_0000 | Main peripherals |
| 0x7C00_0000 | 0xFC00_0000 | PCIe / xHCI |
| 0xFF80_0000 | 0x6000_0000 | ARM local peripherals |

## Files

- `Cargo.toml` — depends on `ruvix-types` + `ruvix-hal` + `ruvix-drivers`. Features: `rpi4`, presumably `rpi5`.
- `README.md` — public docs.
- `src/` — see `src/CLAUDE.md`.
