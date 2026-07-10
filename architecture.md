# MetalOpticalPhoton Architecture

## 1. System Overview

MetalOpticalPhoton is an extension module for the TOPAS Monte Carlo framework that runs optical
photon propagation on an Apple Metal GPU. Geant4 tracks the charged particles and, instead of
creating photons at every step, emits **gensteps** that record only "where, how many, and with what
distribution" to produce photons; the GPU consumes those gensteps and generates, propagates, and
scores the photons.

### Execution flow

```
1. TOPAS initialization — TsGPUOpticalPhysicsModule::ConstructProcess()
   ├── MetalOpticalEngine::Initialize()
   │     ├── create Metal device + command queue
   │     └── compile 5 compute pipelines
   │           (generateScintillationPhotons / generateCerenkovPhotons /
   │            generateBeamPhotons / propagatePhotons / ddaScoring)
   └── BuildGeometry()
         ├── traverse geometry → triangle mesh (TriangleAttr, 32 B) + analytic-solid registration
         ├── collect material properties → MaterialLUT (5 arrays × 256 entries)
         ├── collect surface  properties → SurfaceLUT  (6 arrays × 256 entries)
         └── build BVH acceleration structure (MTLPrimitiveAccelerationStructure; empty BVH if no mesh)

2. Event loop (CPU primary tracking, photon stacking disabled)
   └── per step:
       ├── TsGPUGenstepCollectorProcess::PostStepDoIt()
       │     └── collect scintillation (edep > 0) / Cerenkov (charge ≠ 0) gensteps
       │         (appended lock-free to a thread_local vector)
       └── on batch completion (default 5 events):
           └── FlushBatch()
               ├── HarvestPendingGPUResults() — collect the previous batch's async results
               │     └── each scorer's harvest callback → dispatch the ddaScoring kernel
               └── Propagate() — commit GPU work asynchronously (no waitUntilCompleted)
                   ├── gensteps → photon generation (generate* kernels)
                   ├── propagatePhotons kernel dispatch (HW RT + physics + transit hits)
                   └── globalPhotonOffset += totalPhotons (per-batch RNG uniqueness)

3. Result collection (lazy synchronization)
   └── on GetHits() / GetTransitHits() / GetLastTransitCount()
       └── WaitForGPU() — waitUntilCompleted only when needed
```

### CPU–GPU pipelining

```
CPU: [prep Batch1] → [prep Batch2 + Batch1 results] → [prep Batch3 + Batch2 results] → ...
GPU:                 [propagate Batch1] ───────────→ [propagate Batch2] ───────────→ ...
```

`Propagate()` commits the command buffer without `waitUntilCompleted` and only stores the pending
command buffer, so the CPU's preparation of the next batch overlaps the GPU's propagation (double
buffering). Results are synchronized only when `WaitForGPU()` is reached (e.g. from `GetHits`).

## 2. GPU Kernel Details (`OpticalPhotonKernel.metal`)

### 2.1 Process-competition model

The same "shortest distance wins" model as Geant4:

```metal
float distToAbsorb   = -absLen * log(rand);     // absorption
float distToRayleigh = -rayLen * log(rand);     // Rayleigh scattering
float distToMie      = -mieLen * log(rand);     // Mie scattering
float distToWLS      = -wlsLen * log(rand);     // WLS
float distToBoundary = HW_RT_intersect();       // boundary (HW RT)

float stepDist = min(min(min(distToAbsorb, distToRayleigh),
                         min(distToMie, distToWLS)), distToBoundary);
```

The processes are additionally gated by compile-time `function_constant` flags
(kEnableAbsorption/Rayleigh/Mie/WLS/Boundary, 0–4), so unused branches are removed from the kernel.

### 2.2 Hardware ray tracing

```metal
intersector<> inter;
inter.assume_geometry_type(geometry_type::triangle);
inter.force_opacity(forced_opacity::opaque);
ray r(pos - config.bvhCentroid, dir, 1e-7, 1e10);   // shift by centroid for fp32 precision
auto result = inter.intersect(r, accelStruct);
```

- `tmin = 1e-7`: prevents self-intersection. The ray origin is shifted by `bvhCentroid` to keep
  fp32 precision.
- A retry loop (up to 5 times) rejects grazing hits and re-hits of the previous triangle.
- The BVH is built automatically as an `MTLPrimitiveAccelerationStructure` (an empty BVH is allowed
  for analytic-only setups).

### 2.3 Boundary handling

```
boundary hit → decide matIdOut (match current material against the triangle's inside/outside)
             → hasSurface? (surfaceId valid)
               ├── Yes → ProcessBoundaryWithSurface()
               │         ├── dielectric_metal: reflect/absorb per reflectivity
               │         ├── dielectric_dielectric + reflectivity defined: reflect/refract
               │         └── reflectivity undefined: Fresnel fallback
               └── No  → ProcessBoundaryFresnel()   // pure Fresnel (s/p polarization split)
```

### 2.4 matIdIn/matIdOut decision

The material the photon exits into is decided **not** by the sign of `dir·normal` but by matching
the photon's current material against the triangle's inside/outside materials. This replaced the
`dir·normal` approach because after a reflection/TIR the direction flips and `dir·normal` points to
the wrong side.

```metal
uint matIdIn = photon.materialId;              // = currentMatId
uint matIdOut;
if (currentMatId == hitTri.materialIdInside &&
    currentMatId == hitTri.materialIdOutside)  matIdOut = currentMatId;         // both sides equal
else if (currentMatId == hitTri.materialIdInside)  matIdOut = hitTri.materialIdOutside;
else if (currentMatId == hitTri.materialIdOutside) matIdOut = hitTri.materialIdInside;
else continue;                                 // hit whose side does not match → skip
```

- `materialIdInside/Outside` are fixed when the triangle is registered (`TsGPUOpticalPhysics`).
- The outward normal comes from G4 `GetSurfaceNormal()` (tessellated) or a centroid-based flip
  (polyhedron).

### 2.5 DDA scoring

DDA fluence scoring is **not** inside `propagatePhotons`; it is a **separate `ddaScoring` kernel**
that each scorer's harvest callback dispatches from within `HarvestPendingGPUResults`. The segment
connecting a transit hit's entry/exit points is walked (DDA) across the scoring volume's voxel grid,
accumulating per-bin path length into a per-scorer accumulating GPU buffer.

```
transit hit (entry/exit) → DDA traversal → accumulate per-bin path length → read once at EndOfRun
Fluence = Σ(path length) / bin volume
```

## 3. Triangle Mesh Extraction (`TsGPUOpticalPhysics.cc`)

### G4TessellatedSolid (STL mesh)
```cpp
// direct facet extraction — GetSurfaceNormal() gives the authoritative outward normal
for (int fi = 0; fi < tess->GetNumberOfFacets(); fi++) {
    G4VFacet* facet = tess->GetFacet(fi);
    // 3 triangle vertices + normal
}
```

### Non-analytic solids (CreatePolyhedron + centroid flip)
```cpp
// Geant4 CreatePolyhedron() → G4Polyhedron → triangle decomposition
// flip the normal using the centroid
G4ThreeVector centroid = (v0 + v1 + v2) / 3.0;
G4ThreeVector toCenter = volCenter - centroid;
if (normal.dot(toCenter) > 0) normal = -normal;  // points inward → flip
```

This mesh path is the **fallback** for solids that do not qualify for analytic registration —
e.g. `G4Cons`/`G4Trd`, a rotated `G4Box`, a same-material `G4Box`, or a partial-φ / non-z-aligned
`G4Tubs`. In the common case, an axis-aligned `G4Box` and a z-aligned full-φ `G4Tubs` (together with
`G4Sphere`, `G4Torus`, `G4Polyhedra`) are registered as **analytic** and bypass CreatePolyhedron,
being passed to the kernel through dedicated geometry buffers (§2.2, §4).

## 4. Data Structures

### GPU buffer bindings (`propagatePhotons` kernel)

| Index | Name | Type | Address space |
|---|---|---|---|
| 0 | photons | `Photon[]` (80 B) | device |
| 1 | materials | `MaterialGPU[]` (PropertyTable) | device |
| 2 | surfaces | `SurfaceGPU[]` | device |
| 3 | config | `SimConfig` | **constant** |
| 4 / 5 | hits / hitCount | `Hit[]` / atomic | device |
| 6 / 7 | triangles / numTriangles | `TriangleAttr[]` (32 B) | device |
| 8 | totalPhotons | uint | device |
| 9 | accelStruct | acceleration_structure | — |
| 10 / 11 / 12 | transitHits / transitCount / maxTransitHits | `TransitHit[]` / atomic | device |
| 13 | diagCounters | `atomic_uint[]` (host buffer 8192) | device |
| 14 / 15 | trajectoryBuf / trajectoryCount | (debug) | device |
| 16 / 17 | matLUTs / surfLUTs | `MaterialLUT[]` / `SurfaceLUT[]` | device |
| 18 | photonMeta | `PhotonMeta[]` (cold SOA) | device |
| 19, 21–30 | torus / box / sphere / cylinder / polyhedra / surface analytic geometry | device |

Global constant: `MOP_MAX_PHOTONS_PER_BATCH = (1 << 27)` ≈ 128 M (photon-buffer capacity /
per-dispatch cap). `FlushBatch` and BEAM mode split and dispatch separately in **5 M (5,000,000)
chunks**.

### Photon struct (80 B) + PhotonMeta (cold)

```metal
struct Photon {                 // 80 B
    packed_float3 position;     // 12 B
    packed_float3 direction;    // 12 B
    packed_float3 polarization; // 12 B
    float wavelength, energy, time, weight;                     // 16 B
    uint status, volumeId, materialId, stepCount, flags;        // 20 B
    uint reflectedCount;                                        // 4 B
    int  lastHitTriId;                                          // 4 B
};
struct PhotonMeta { uint genstepId; uint parentTrackId; };      // separate cold buffer (18)
```

`genstepId`/`parentTrackId` were split out of `Photon` into a separate cold `PhotonMeta` buffer
(buffer 18) for a hot/cold layout. The CPU-side `MOPPhoton`/`MOPPhotonMeta` mirror the same layout.

### PropertyTable

```metal
struct PropertyTable {
    uint count;                             // ★ first field (offset 0)
    float energies[MAX_PROPERTY_ENTRIES];   // MAX_PROPERTY_ENTRIES = 512
    float values[MAX_PROPERTY_ENTRIES];
};
```

A GPU copy of the original G4 material property table (energy-based interpolation).

### MaterialLUT (`LUT_SIZE = 256`)

```metal
struct MaterialLUT {
    float rindex[256];   // refractive index
    float absLen[256];   // absorption length — ★ not a uniform grid; packs G4 not-a-knot spline knots
    float rayLen[256];   // Rayleigh scattering length
    float mieLen[256];   // Mie scattering length
    float wlsLen[256];   // WLS absorption length
};
```

The 1.0–15.0 eV range is prefilled on a 256-point uniform grid, replacing the `InterpolateProperty`
binary search with `InterpolateLUT` direct indexing (2 adjacent reads). The **`absLen` slot is the
exception** — for sub-1% CPU/GPU agreement it stores the G4 not-a-knot spline coefficients
(`[0]=N, [1..64]=E, [65..128]=V, [129..192]=D2`, `MAX_KNOTS = 64`) and is evaluated per-photon by
`SplineEvalG4Packed` (binary search).

### SurfaceLUT (`LUT_SIZE = 256`)

```metal
struct SurfaceLUT {
    float reflectivity[256];   // reflectivity
    float efficiency[256];     // detection efficiency
    float transmittance[256];  // transmittance
    float specularLobe[256];   // UNIFIED lobe probability
    float specularSpike[256];  // UNIFIED spike probability
    float backScatter[256];    // UNIFIED backscatter probability
};
```

### TriangleAttr

```metal
struct TriangleAttr {           // 32 B (compressed from the 64 B Triangle)
    packed_float3 normal;       // 12 B — outward normal
    uint volumeId;              // 4 B
    uint materialIdInside;      // 4 B
    uint materialIdOutside;     // 4 B
    uint surfaceId;             // 4 B
    uint _pad;                  // 4 B (32 B alignment)
};
```

Vertex data (v0, v1, v2) is managed by the Metal HW RT BVH itself, so it is excluded from
TriangleAttr.

## 5. Scoring

### GPU (`TsScoreGPUOpticalPhotonFluence`)
- DDA ray tracing along the segment joining a transit hit's entry/exit points → per-bin path length.
- Each scorer registers a harvest callback and, from within `HarvestPendingGPUResults`, dispatches
  the `ddaScoring` kernel into a per-scorer accumulating GPU buffer (FIFO-ordered ahead of the next
  propagation, so it reads the correct batch's hits).
- The buffer is copied once into `fFirstMomentMap` at EndOfRun. `Fluence = Σ(path_length) / bin_volume`.
- **Multi-scorer**: all scorers register callbacks, then the last one issues a single
  `ForceRepropagate`, and each scorer's DDA reuses that single propagation's transit hits
  (deferred-trigger).

### CPU (`TsScoreFluence` — TOPAS built-in)
- Computes `GetStepLength()`, divides it by the bin volume (`GetCubicVolume`), multiplies by the
  pre-step weight, and accumulates that value into the single bin returned by
  `fComponent->GetIndex(aStep)`.
- Same physical result: it is normalized by bin volume just like the GPU (so the totals match); the
  only difference is that the GPU distributes along a DDA path across several bins.

## 6. RNG (Random Number Generator)

A port of CLHEP `RanecuEngine` (L'Ecuyer 1988 combined MRG):

```metal
struct RandomState { int seed1; int seed2; };
// RandUniform: the combined-MRG recurrence with L'Ecuyer constants (ecuyer_a = 40014, ...)
// InitRandomState(photonId, globalSeed, kernelSalt): initializes the two seeds via a splitmix32 hash
```

- Each photon gets an independent RNG stream from `idx + globalPhotonOffset`; per-batch uniqueness is
  maintained by `globalPhotonOffset += totalPhotons`.
- GPU thread-safe (no stream collisions). No cipher-based RNG (e.g. TEA) is used.

## 7. Lifecycle / Batching

### 7.1 Tail-batch recovery

Removes the case where the last incomplete batch is not flushed when `EventBatchSize` does not divide
the total event count:

```
TOPAS event loop ends
  ↓
TsScoreGPUOpticalPhoton{Fluence,…}::UserHookForEndOfRun()
  ↓
TsGPUOpticalPhysics::FinalizePendingBatch()
  ├── FlushBatch()               — send any pending batched gensteps to the GPU
  └── HarvestPendingGPUResults() — collect the last result from the async pipeline
  ↓
copy the per-scorer accumulating GPU buffer → fFirstMomentMap → Output()
```

Residual transit hits are already accumulated into the per-scorer GPU buffer by the harvest
callback, so EndOfRun simply reads that buffer (the older per-event CPU-DDA/`fEvtMap` path is kept
for diagnostic counts only, to avoid double counting).

### 7.2 Use-after-free prevention

The `TsGPUOpticalPhysicsModule` destructor nulls every scorer's static `fGPUEngine` pointer **before**
deleting the shared engine:

```cpp
TsScoreGPUOpticalPhotonCount::SetGPUEngine(nullptr);
TsScoreGPUOpticalPhotonFluence::SetGPUEngine(nullptr);
TsScoreGPUOpticalPhotonSurfaceTrackCount::SetGPUEngine(nullptr);
TsScoreGPUOpticalPhotonPhaseSpace::SetGPUEngine(nullptr);
delete g_sharedGPUEngine;  g_sharedGPUEngine = nullptr;
```

Later hooks detect the nullptr and return early. GPU dispatch is serialized by a recursive mutex
(`m_gpuMutex`) to allow the harvest callback to re-enter.

## 8. Design Decision: No Variance Reduction

This system **does not use variance reduction**. Even for geometries with very low detection
efficiency, brute force (increasing the event count) was found to be optimal.

| Technique | Reason for rejection |
|------|----------|
| Shell / Volume / Weight-Window IS | correlation ρ ≈ 1 → N_eff ≈ 1, FOM worsens |
| Direction RR / Cone biasing | weight spikes vs bias trade-off at high resolution |
| Adjoint MC (backward) | reconstructing the spatial distribution needs a high-dimensional tensor; cannot replace forward |
| Post-lens splitting | deterministic refraction → copies are not decorrelated |
| Survival biasing | gain negligible |

Checklist before adopting any future VR idea: ① estimate the correlation coefficient ρ ② check for
weight spikes in high-resolution bins ③ whether the spatial distribution can be reconstructed
④ measure FOM against the baseline.
