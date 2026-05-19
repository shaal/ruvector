# ruvector-hailo-cluster/deploy

Operator deployment assets for the Hailo cluster, mmWave radar bridge, RuView CSI bridge, ruvLLM bridge, and ruvLLM Pi worker. These are not consumed by `cargo build`; they are shipped to the Pi fleet by the install scripts.

## Systemd units + envs

- `ruvector-hailo-worker.service` + `ruvector-hailo.env.example`
- `ruvector-mmwave-bridge.service` + `ruvector-mmwave-bridge.env.example`
- `ruview-csi-bridge.service` + `ruview-csi-bridge.env.example`
- `ruvllm-pi-worker.service` + `ruvllm-pi-worker.env.example`
- `ruvllm-bridge.env.example`

## udev rules

- `99-hailo-ruvector.rules` — grants access to `/dev/hailo0`.
- `99-radar-ruvector.rules` — grants access to mmWave `/dev/ttyUSB*` nodes.

## Install scripts

- `install.sh` — top-level install.
- `install-mmwave-bridge.sh`, `install-ruview-csi-bridge.sh`, `install-ruvllm-bridge.sh`, `install-ruvllm-pi-worker.sh` — per-service installers.

## Build / cross-build

- `cross-build.sh`, `cross-build-bridges.sh` — aarch64 cross-builds of worker and bridge binaries.
- `setup-hailo-compiler.sh` — bootstraps the Hailo Dataflow Compiler toolchain.
- `compile-hef.sh`, `compile-hef.py`, `compile-encoder-hef.py` — turn ONNX into Hailo HEFs.
- `download-cpu-fallback-model.sh`, `download-encoder-hef.sh` — model artefact fetchers.
- `export-minilm-onnx.py`, `export-minilm-encoder-onnx.py` — exporters used as input to HEF compilation.
- `ruvllm-cluster-smoke.sh` — end-to-end smoke test against a running fleet.

See `../CLAUDE.md`.
