# Uninstall and Reset (Windows-first)

Use this guide to stop Aurora-Lens, export diagnostics, and remove local runtime data.

## Stop and verify

1. `aurora-lens stop`
2. `aurora-lens status`

## Export diagnostics

Run:

```powershell
aurora-lens support-export
```

This writes a support bundle under the deterministic runtime support directory.

## Remove local runtime files

Delete the Aurora-Lens runtime root shown by `aurora-lens status`:

- state files (PID/ownership)
- logs
- temporary runtime files
- support bundles

## Uninstall package

```powershell
pip uninstall aurora-lens
```

For full cleanup, also deactivate and delete the virtual environment used for install.
