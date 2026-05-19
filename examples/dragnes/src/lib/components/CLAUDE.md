# dragnes / src / lib / components

Reusable Svelte 5 components used by the DrAgnes UI.

## Important files
- `DermCapture.svelte` - camera / image capture component for dermoscopic input.
- `ClassificationResult.svelte` - renders the classifier output and per-class probabilities.
- `ABCDEChart.svelte` - chart of the ABCDE dermoscopic scores from `$lib/dragnes/abcde.ts`.
- `GradCamOverlay.svelte` - Grad-CAM heatmap overlay over the lesion image.
- `LesionTimeline.svelte` - longitudinal view of a lesion's previous captures.
- `DrAgnesPanel.svelte` - composite side-panel that orchestrates the above.

## Related
- Logic backing these components: `../dragnes/`. Mounted from routes in `../../routes/`.
