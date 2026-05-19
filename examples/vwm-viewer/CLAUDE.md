# vwm-viewer

Static, self-contained HTML viewers for the RuVector VWM ("Volumetric World Model" / 4D Gaussian splatting) visualizations. Pure HTML/JS - no build step. Includes a Docker/nginx wrapper for trivial deployment.

## Files

- `canvas-viewer.html` (~15 KB) - Canvas2D 4D Gaussian splatting viewer with stats panel, coherence badge, and a transport bar / time scrubber.
- `football.html` (~84 KB) - Larger demo (sports-themed VWM scene).
- `Dockerfile` - Minimal `nginx`-based container that serves this folder.
- `nginx.conf` - nginx config used by the Dockerfile.

## How to run

```bash
# Direct: open in a browser
xdg-open /home/user/ruvector/examples/vwm-viewer/canvas-viewer.html

# Or via Docker:
cd /home/user/ruvector/examples/vwm-viewer
docker build -t vwm-viewer . && docker run --rm -p 8080:80 vwm-viewer
```

## Tech stack

- Plain HTML5 + Canvas2D + vanilla JS. No bundler, no dependencies.
- Optional: nginx (Dockerfile).

## Related

- RVF dashboard with 3D scenes: `examples/rvf/dashboard/`.
- WASM-backed examples: `examples/onnx-embeddings-wasm`, `examples/wasm/ios`, `examples/scipix/web`.
