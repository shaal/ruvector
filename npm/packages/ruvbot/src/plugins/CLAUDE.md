# ruvbot / src / plugins

Plugin system. Lets third parties extend ruvbot at runtime with new
skills, channel adapters, or providers.

## Files
- `PluginManager.ts` - Loads/activates/deactivates plugins, validates
  their manifests, and surfaces them to the rest of the framework.
- `index.ts` - Barrel exporting the manager and plugin contract.
