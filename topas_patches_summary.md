# OpenTOPAS source patches (vs vanilla)

The only patch this build applies to the vanilla OpenTOPAS **source is a single
`topas_patches/opentopas_local.patch`** (4 files). Its **core is two changes** — a
parallel-world envelope volume shrink and a primary-polarization typo fix — both of which
directly affect CPU optical accuracy, so they matter even to CPU-only users who never touch the GPU.

It is applied automatically at build time: a marker string (`MetalOpticalPhoton`) is used to skip
re-applying, then it is gated by `patch -p1 --forward --dry-run`, applied, and all 4 marker files are
verified. **Required versions: OpenTOPAS ≥ 4.2.3 + Geant4 ≥ 11.3.2** (the patch targets 4.2.3).

> The GPU photon-propagation engine itself (its extension module, performance optimizations, analytic
> geometry, GPU-scorer safeguards, etc.) is **not** a patch to vanilla — it is added code. See
> [`architecture.md`](architecture.md) for its structure and behavior.

---

## Patch 1 — parallel-world envelope volume shrink (`geometry/Ts{Box,Cylinder,Sphere}.cc`)

**Symptom**: when a parallel-world (`IsParallel=True`) component's envelope coincides exactly with a
mass-world component, G4PathFinder's multi-navigator min-step coordination returns step = 0. A
grazing-incidence photon loops on the same boundary indefinitely → max-step abort → the photon
vanishes. The run finishes quickly, but the fluence/edep are heavily (tens of %) under-counted —
silently, from the user's point of view.

**Fix**: when `IsParallel()`, auto-shrink the envelope inward by 1 nm.

```cpp
if (IsParallel()) {
    const G4double pwAutoShrink = 1.0 * CLHEP::nm;
    totalHLX -= pwAutoShrink;  totalHLY -= pwAutoShrink;  totalHLZ -= pwAutoShrink;  // TsBox
}
```

- `TsBox.cc`: shrink HLX/HLY/HLZ (+ add `#include "G4SystemOfUnits.hh"`).
- `TsCylinder.cc`: shrink RMax·HL, expand RMin inward.
- `TsSphere.cc`: shrink RMax, expand RMin.

> **Timing characteristic**: after the patch the CPU run takes longer — because the previously-stuck
> photons are no longer aborted and instead transit correctly to the end (a physically correct result
> with no losses).
>
> **Bundled narrow fix**: `TsSphere.cc` also carries a `GetIndex()` rewrite besides the envelope
> shrink — for a spherical scorer divided into several concentric radial shells, the G4 navigator's
> replica number gives the wrong R bin; the fix transforms the PreStep position into envelope-local
> coordinates and computes the (r, φ, θ) bin index directly (only relevant when using a
> multi-division spherical scorer).

---

## Patch 2 — primary-opticalphoton polarization typo fix (`primary/TsVGenerator.cc`)

**Symptom**: when `BeamPolarization` is unset (= unpolarized/random pol), the upstream code tries to
build a random transverse polarization with a cross product, but has two defects that depend on the
beam direction:

- **Off-axis beam — typo**: the `else` branch's final assignment is `fPolY = polarization.z();`, so
  `fPolZ` is never set and `fPolY` gets overwritten (and the base polarization in this branch is not
  even perpendicular to the direction) → the polarization becomes invalid. Oblique-incidence /
  polarized cases (e.g. the U1 Brewster p-pol test) fall here.
- **Axis-aligned beam — fixed polarization**: when `tanTheta == 0` (an xy-plane beam with `dCos3=0`,
  or a pure ±z beam), the polarization is pinned to `(1,0,0)`. For ±z·±y beams this is a valid
  transverse vector but it is **fixed, not random**, so unpolarized photons lose their azimuthal
  randomness and the result is **biased** (the Fresnel calculation itself is not broken). For a ±x
  beam, `(1,0,0)` is parallel to the direction → a **longitudinal (invalid)** polarization.

In short, unpolarized/random-pol primary photons do not receive a correct random transverse
polarization, so — depending on beam direction — the Fresnel/TIR result is either biased (axis-aligned)
or broken (off-axis).

**Fix**: clean up the block so that **`fPolZ` is set correctly** (= the known Y→Z typo fix) and remove
the `tanTheta==0` special case. From an orthonormal basis (u, v) perpendicular to the direction,
`polarization = cos φ·u + sin φ·v` gives a random transverse polarization for every direction.

```cpp
// key: the last assignment is fPolZ, not fPolY
fPolX = polarization.x();
fPolY = polarization.y();
fPolZ = polarization.z();   // upstream typo: fPolY = polarization.z();
```

**Impact**: normalizes the Fresnel R/T and TIR for every unpolarized/random-pol primary
opticalphoton. Affects both CPU and GPU (at the photon-source stage).

> **Bundled narrow fix**: another hunk in the same file removes the forced 0-eV bin prepend for
> continuous energy spectra — previously a 0 eV bin was prepended, stretching the sampling support
> down to 0…E_min and mixing in a low-energy triangular-ramp outlier; this pins the support to the
> user's `Values` [E_min, E_max] (only relevant when using a continuous-spectrum beam).

---

## Summary

The only patch to vanilla OpenTOPAS is `opentopas_local.patch` (4 files), and its core is two changes:

1. **Parallel-world envelope 1 nm shrink** (`Ts{Box,Cylinder,Sphere}.cc`) — without it, grazing
   photons hit a stuck loop and CPU photons are silently lost.
2. **TsVGenerator polarization typo fix** (`primary/TsVGenerator.cc`) — `fPolY = .z()` → `fPolZ`
   (the Y→Z fix).

Two narrow fixes are bundled in the same files (TsSphere `GetIndex()`, continuous-spectrum 0-eV bin).
Everything else related to the GPU photon engine is extension code, not a patch, and is covered in
[`architecture.md`](architecture.md).
