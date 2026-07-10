# TOPAS source patches

The OpenTOPAS source patch that `build_topas_gpu.sh` **applies automatically** before building.
It is required because building vanilla OpenTOPAS as-is produces wrong optical-photon results.

> **License**: this patch is a modification of OpenTOPAS source files
> (TsBox/TsCylinder/TsSphere/TsVGenerator) and remains under OpenTOPAS's **MIT License** — keep the
> OpenTOPAS copyright/MIT notice when applying or redistributing. For the full third-party notices
> (including Geant4), see [`../NOTICE`](../NOTICE) at the repo root.

## Target versions

- upstream: `https://github.com/OpenTOPAS/OpenTOPAS`
- **Supported: OpenTOPAS `v4.2.3` or later + Geant4 `11.3.2` or later.**
- The build script **aborts the build** if OpenTOPAS is below `v4.2.3` (or Geant4 below `11.3.2`).
  After the version gate passes, it checks right before applying that the patch applies cleanly with
  `patch --forward --dry-run`. If that fails, the patch is **not** applied and the script prints a
  warning plus the manual command — **the build itself continues on the vanilla source**, so in that
  case you must patch manually and rebuild (an unpatched build gives wrong optical results).

## The patch file (a single `opentopas_local.patch`)

One vanilla patch. It modifies 4 files:

| File | Change | Without it |
|---|---|---|
| `primary/TsVGenerator.cc` | ① remove the forced 0 eV zero-bin prepend for continuous spectra  ② random-polarization typo fix — the final assignment `fPolY = polarization.z();` leaves `fPolZ` unset (the Y→Z fix) + remove the `tanTheta==0` fixed-polarization special case (the block is cleaned up into an Opticks-style u,v basis giving a random transverse polarization for every direction) | ① a triangular-ramp outlier over 0…E_min  ② off-axis beams: invalid polarization → Fresnel/TIR errors; ±z beams: fixed (1,0,0) polarization → up to ~40 % shift of the in-plane Rayleigh halo |
| `geometry/TsBox.cc` | auto-shrink the parallel-world envelope by 1 nm per axis (`IsParallel()`) + add `G4SystemOfUnits.hh` include | PW boundary coincident with the mass world → G4Navigator stuck loop → stuck photons hit the max-step abort (silent photon loss; fluence/edep under-counted) |
| `geometry/TsCylinder.cc` | same auto-shrink (RMax/HL −1 nm; if RMin>0, expand it inward by +1 nm) | 〃 |
| `geometry/TsSphere.cc` | same auto-shrink + fix of the nested concentric R-division wrong-copyNo bug in `GetIndex()` | 〃 + a sphere R-division scorer accumulates into wrong bins |

Issue-by-issue write-up: [`../topas_patches_summary.md`](../topas_patches_summary.md).

## Automatic application (build_topas_gpu.sh)

1. If the marker is present in BOTH `TsBox.cc` (geometry) and `TsVGenerator.cc` (polarization) →
   **already applied, skip**.
2. Otherwise → apply with `patch -p1 --forward` (validated first with a dry-run; then all 4 marker
   files are verified).
3. If application fails (version mismatch etc.) → warning + manual-command instructions
   (the build continues unpatched — see above).

## Manual apply / revert

```bash
# apply
patch -p1 -d /path/to/OpenTOPAS < opentopas_local.patch
# revert
patch -p1 -R -d /path/to/OpenTOPAS < opentopas_local.patch
```

## Regenerating the patch (after further edits to the TOPAS source)

```bash
# run from the MetalOpticalPhoton repo root (the output path is CWD-relative)
git -C /path/to/OpenTOPAS diff -- \
    primary/TsVGenerator.cc geometry/TsBox.cc geometry/TsCylinder.cc geometry/TsSphere.cc \
    > topas_patches/opentopas_local.patch
```
