# MetalOpticalPhoton

> **A TOPAS/Geant4 optical-photon simulation engine that keeps Geant4's optical physics intact
> and offloads only the heavy photon propagation to the Apple Metal GPU.**

## Overview

In Monte Carlo simulation of optical detectors — a scintillator feeding mirrors, lenses, and a
camera — the behavior of the optical photons themselves decides the result. Geant4 (and TOPAS on
top of it) solves this optical physics exactly — Fresnel reflection, Snell refraction, Rayleigh/Mie
scattering, wavelength shifting (WLS), boundary-surface reflection — but a single charged particle
produces tens to hundreds of thousands of photons, each created and tracked as an individual
`G4Track`, so photon tracking (`g4optical`) dominates the CPU time of a realistic detector setup.

The key observation is simple: each photon propagates independently of every other, and the work
reduces to one operation — ray tracing through the geometry. That is exactly what GPUs, and in
particular Metal ray tracing on Apple Silicon (hardware-accelerated on M3 and later), do best. But
moving the work is not the goal by itself: the acceleration only means something if **no physical
accuracy is given up**.

So the engine splits the roles. **Geant4 keeps the domain where it is authoritative — charged-
particle transport and energy deposition** — but instead of creating photons at every step it emits
**gensteps** that record only "where, how many, and with what distribution". The Metal kernel
receives those genstep batches, expands them into photons, and propagates them through the geometry
— meshes via the BVH acceleration structure, registered solids via analytic intersection — applying
the same boundary physics as Geant4 all the way to the end.

In short, **the optical physics stays line-by-line identical to Geant4** while only the heavy
photon propagation moves to the GPU — the aim is to cut propagation time without bargaining away
accuracy. The build also patches vanilla-OpenTOPAS bugs that affect optical accuracy, which matter
even to CPU-only users who never touch the GPU.

## Key features

- **Optical-physics parity**: Fresnel/Snell, Rayleigh/Mie, UNIFIED surface, WLS, per-photon
  polarization — Geant4-equivalent implementations.
- **Analytic geometry**: Box/Sphere/Cylinder/Torus/Polyhedra are solved analytically instead of via
  meshes, eliminating the inscribed-polygon bias of the curved solids (Sphere/Cylinder/Torus).
- **Vanilla OpenTOPAS patches**: bugs affecting CPU optical accuracy (parallel-world envelope stuck
  loop, primary-polarization typo) are fixed automatically at build time.
- **GPU BEAM mode**: opticalphoton beam sources generated kernel-side on the GPU — for mass photon
  production (1B+).

Details: [`architecture.md`](architecture.md) (system architecture) and
[`topas_patches_summary.md`](topas_patches_summary.md) (patches vs vanilla).

## Quick start

### Build

```bash
cd MetalOpticalPhoton
./build_topas_gpu.sh        # path auto-detection + TOPAS patch + shader/dylib/TOPAS build + install
```

Prerequisites (Xcode + Metal toolchain / Homebrew / **OpenTOPAS 4.2.3+ + Geant4 11.3.2+** / GDCM /
**Qt6**) and the build modes/options are covered in [`INSTALL.md`](INSTALL.md). The patch is applied
automatically during the build — but on failure (unsupported version etc.) the build continues on
vanilla source with only a warning, so check the patch stage (4/4 verification) in the build log;
see [`topas_patches/README.md`](topas_patches/README.md).

### Run

```bash
# the patched binary installed by the build (location: INSTALL.md; just `topas-gpu` if on PATH)
<OpenTOPAS-install>/bin/topas-gpu your_sim.txt
```

Enable the GPU optical module in your TOPAS script:

```
s:Ph/ListName = "Optical"
sv:Ph/Optical/Modules = 2 "g4em-standard_opt4" "gpuoptical"
```

Validation examples: [`benchmark/`](benchmark/) 01–10.

### Requirements

- macOS + Apple Silicon (M series — a Metal-ray-tracing-capable GPU)
- Xcode + the Metal toolchain (Metal 3.1+; Command Line Tools alone are not enough —
  [`INSTALL.md`](INSTALL.md) §1-1)
- CMake 3.20+, Qt6
- OpenTOPAS 4.2.3+ + Geant4 11.3.2+

## Documentation

| Doc | Contents |
|---|---|
| [`architecture.md`](architecture.md) | System architecture (CPU–GPU pipeline, memory layout) |
| [`topas_patches_summary.md`](topas_patches_summary.md) | Source patches vs vanilla OpenTOPAS (CPU accuracy: envelope shrink + polarization typo) |
| [`INSTALL.md`](INSTALL.md) | Installation (automatic/manual), version requirements |
| [`benchmark/`](benchmark/) | Validation examples 01–10 |

## License

MetalOpticalPhoton is under the **MIT License** ([`LICENSE`](LICENSE)). As a TOPAS extension it is
built on and distributed with OpenTOPAS (MIT) and Geant4 (Geant4 Software License); the copyright,
notice, and citation requirements of both third-party packages are collected in
[`NOTICE`](NOTICE) — keep the NOTICE statements (in particular the Geant4 Collaboration
acknowledgement) when redistributing, and cite Geant4 and TOPAS/OpenTOPAS together when publishing
results obtained with this software.
