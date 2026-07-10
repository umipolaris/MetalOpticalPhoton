# 10 — Curved-surface geometry analytic vs mesh

Compares the ray/surface intersection accuracy of curved solids (sphere, cylinder, torus)
between GPU (Metal) and Geant4 CPU. The GPU engine can intersect a curved surface via two
paths — **analytic** (solving the equation directly in the kernel: sphere and cylinder use a
quadratic, torus uses sphere-tracing bracket + bisection on the exact implicit surface) and **mesh** (triangulating the solid
and traversing the hardware BVH, identical to STL). This benchmark verifies that **each** of the
two paths reproduces the CPU Geant4 solid and that they agree with each other, for all three
curved primitives. The path selection is done **at the script level without recompiling**, via
the `GPUForceMesh` parameter.

(No intro figure — the deliverable of this benchmark is the chord table from `analyze.py`, not a plot.)

## Setup

The core of the measurement design is to isolate and measure **geometry intersection accuracy
only** — it excludes the influence of boundary optics (Fresnel/Snell), scoring discretization, and
detector geometry. The cleanest probe is **bulk absorption along a known chord**. Filling the solid
with a non-scattering, non-refracting absorber (n=1, absorption length `Labs = 10 mm`) means there is
**no** refraction or reflection at the surface because n=1, and the only thing the surface does is
define the start and end of the absorption region along the ray. The intersection-distance error
therefore translates directly into the absorption path-length (chord) error.

| solid | object | beam direction | chord |
|---|---|---|---|
| sphere | `TsSphere` RMax=R, centered at origin | +Z, on axis | 2R |
| cylinder | `TsCylinder` RMax=R, axis Z, centered | +X (⊥ to axis) | 2R |
| torus | `G4Torus` RMax=R, RTor=5 cm, full | +Z, through tube | 2R |

For the cylinder, instead of rotating the cylinder itself, the source group is rotated with
`RotY=-90 deg` to make the beam point +X — because rotating the cylinder off the Z axis would by
itself force the mesh path and break the analytic comparison.

### Geometry / materials

| Volume | Type | dimensions | material |
|---|---|---|---|
| `World` | `TsBox` | HLX=HLY=HLZ=30 cm | Air |
| `Obj` (curved) | `TsSphere` / `TsCylinder`(HL=8 cm) / `G4Torus`(RTor=5 cm) | RMax=R ∈ {1,2,3} cm | Air (transparent) or `AbsMat` (absorbing) |
| `Det` | `TsBox` | HLX·HLY·HLZ 0.5×2×2 cm (thin face rotated depending on axis) | Air |

- `Air` (world / transparent run): RIndex 1.0 (flat), AbsLength 1e6 cm (non-absorbing).
- `AbsMat` (absorbing run): Hydrogen, ρ=0.001 g/cm³, RIndex 1.0, **AbsLength 1.0 cm = `Labs`**.
  Since n=1 there is no refraction/reflection, only absorption acts.

### Source / Scorer

- **Source**: `OptSrc` Beam, `opticalphoton`, 2.5 eV, `BeamPositionDistribution=Flat` · Rectangle
  cutoff **0.5 × 0.5 mm** pencil beam. `OptSrcGroup` is placed at −20 cm so the beam passes through
  the center of the solid, and only the cylinder uses `RotY=-90 deg`. Photon count defaults to
  **2,000,000** histories. The GPU also uses the TOPAS-Beam source path (not the BEAM kernel).
- **Scorer**: `Det` with 1×1×1 bin. GPU `Quantity="GPUOpticalPhotonFluence"`, CPU `Quantity="Fluence"`.
  (Not parallel-world; scoring directly onto the mass-world detector volume.)

**Measurement metric — chord `L`.** Since absorption is exponential, the GPU-internal chord is
directly recovered from the surviving fluence at the downstream detector
`F_abs = F0·exp(-L/Labs)`:

```
L_GPU = -Labs · ln(F_abs / F0)
```

`F0` is taken from the transparent run (absorber→Air) of the **same solid and same engine**, which
cancels the detector solid angle, binning, and per-engine normalization (hence the transparent
baseline is measured per engine, and for the GPU per path). The chord is the trusted metric; the
raw transmission ratio `F_GPU/F_CPU` is also reported, but as the transmission drops (at R=3 cm the
chord is 60 mm, survival ~0.25 %) it scatters at the percent level from Poisson noise — read the
chord.

## Execution

| script | output (`results/`) | method |
|---|---|---|
| `run.sh` | `<solid>_<analytic\|mesh>_R<r>_G_s<seed>.csv` (GPU), `<solid>_R<r>_C_s<seed>.csv` (CPU), `*_trans.csv` (transparent baseline) | 3 solids × R{1,2,3} cm × {analytic, mesh, CPU} × 5 seeds, config emitted inline |
| `analyze.py` | (chord table on stdout) | analytic & mesh GPU chord vs CPU & true (2R) |

```bash
cd benchmark/10_analytic_vs_mesh
bash run.sh           # run all cases (2e6 histories each)
python3 analyze.py    # print chord table: GPU-analytic & GPU-mesh vs CPU·true
```

analytic ↔ mesh differ only by whether the emitted config includes or omits the
`b:Ge/Obj/GPUForceMesh="True"` (mesh) line. The same toggle drives sphere, cylinder, and torus
identically (the env vars `MOP_SPHERE_ANALYTIC`/`MOP_CYLINDER_ANALYTIC` still exist for
sphere/cylinder and are OR-combined with `GPUForceMesh`; `GPUForceMesh` adds the per-volume
script-level toggle and is the only mesh toggle for the torus).

### Environment variables

| variable | default | scope | description |
|---|---|---|---|
| `HIST` | `2000000` | histories per case | override beam photon count |
| `SEEDS` | `(11 22 33 44 55)` | multi-seed | array inside `run.sh` (transparent baseline fixed to seed 42) |
| `RADII` | `(1.0 2.0 3.0)` | radius scan (cm) | array inside `run.sh` |
| `TOPAS` | patched `topas-gpu` auto-discovered | binary path | resolved by `../common/topas_env.sh`. unpatched forbidden (fluence ~4× over) |

All runs are invoked with `G4TRACE_DIR=OFF` (inside `run.sh`).

## Notes — chord recovery and beam alignment

- **The chord is the canonical metric**, the transmission ratio is for reference. Since the
  surface-intersection distance error is not amplified by `Labs` but appears 1:1 in the chord error,
  sub-mm resolution is possible.
- **The transparent baseline is measured separately per engine and per path.** Using one engine's
  `F0` against another engine's `F_abs` injects a spurious offset.
- The cylinder is hit by rotating the beam to +X (the axis stays Z) — because the analytic cylinder
  path requires the axis to be the global Z.
- For the torus, `Det` and `OptSrcGroup` are shifted by `TransX=5 cm` so the beam passes +Z through
  the center of the `RTor=5 cm` tube.

## File structure

```
10_analytic_vs_mesh/
├── run.sh                # runner: emits config inline (no configs/ directory) + GPU/CPU run
├── analyze.py            # chord table: GPU-analytic & GPU-mesh vs CPU·true(2R)
└── results/
    ├── <solid>_analytic_R<r>_G_s<seed>.csv   # GPU analytic (solid∈{sphere,cylinder,torus}, r∈{1,2,3}, seed∈{11,22,33,44,55})
    ├── <solid>_mesh_R<r>_G_s<seed>.csv       # GPU mesh
    ├── <solid>_R<r>_C_s<seed>.csv            # CPU Geant4
    └── <...>_trans.csv                        # transparent baseline (per engine·path, seed 42)
```

The config is not a separate file; `run_one()` in `run.sh` writes it to a temporary `.cfg_<tag>.txt`,
runs it, and deletes it afterward (there is no `configs/` directory). `results/*.csv` is regenerated
with `bash run.sh`.

Master index: [`../README.md`](../README.md)
