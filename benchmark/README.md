# MetalOpticalPhoton Benchmarks

Each directory is **standalone** runnable: run the device-separated `run_cpu.sh` / `run_gpu.sh`
(some variants — see each benchmark's README), then the folder's plot/analysis script (plot.py in most; plot_color.py in 03, scripts/fig_02_angle_scan.py in 02, analyze.py in 10; 01 tabulates timings from its logs). The `topas-gpu` binary is
resolved in the order **`TOPAS` env > `PATH` (`command -v`) > default install path** — for a global
change edit only `common/topas_env.sh` (or `TOPAS=/path ./run_*.sh`). All runs apply
`G4TRACE_DIR=OFF` automatically. CSVs (mostly gitignored, regenerable), logs, and figures land under
each benchmark's `results/` (02 keeps its built configs under `results/built/`); `configs/` holds
the TOPAS `.txt` configs (01 and 05's run scripts symlink the lens `.stl` from common/ at run time; the link is gitignored).

## Installation smoke test

```bash
./run_smoke_test.sh                # all 10 benchmarks, GPU + CPU (~10 min; 04-cpu dominates)
DEVICES=gpu ./run_smoke_test.sh    # GPU only (~2 min)
ONLY="03 07" SCALE=10 ./run_smoke_test.sh   # subset; SCALE multiplies every event count
```

Runs every benchmark once on **drastically reduced event counts** and checks, per run: exit code,
crash-free log, GPU-engine engagement (`[TsGPU]` marker — and its *absence* on CPU rows), and that
the expected output CSVs exist with a nonzero sum. Writes a pass/fail HTML report to
`smoke_logs/smoke_report.html`. This verifies the **installation** end-to-end — not physics
statistics; for real numbers use each benchmark's own scripts below.

## Benchmarks

| # | Directory | Purpose | Key measurement |
|---|---|---|---|
| 01 | [`01_speed/`](01_speed/) | sim_light N-scan speed: standard 1×1×50 + high-res 816×624×50 | GPU vs CPU 30T, N=1K/10K/100K |
| 02 | [`02_reflection_refraction_tir/`](02_reflection_refraction_tir/) | Fresnel reflection/refraction (U1) + total internal reflection TIR (U2) | GPU/CPU physics agreement |
| 03 | [`03_rayleigh/`](03_rayleigh/) | Rayleigh wavelength-selective scattering colour-separation demo (blue sky / red sunset) | colour separation, GPU/CPU sub-0.1% |
| 04 | [`04_birks/`](04_birks/) | Birks scintillation Q(T) (U5_short) | Q value agreement, GPU/CPU sum ratio |
| 05 | [`05_torus/`](05_torus/) | torus + diffuse mirror + lens production benchmark | 10B (GPU single shot / CPU 1e9×10seed), sub-1% |
| 06 | [`06_mie_diffusion_cyl/`](06_mie_diffusion_cyl/) | Mie scattering diffusion (cylinder + fog) | GPU/CPU fluence agreement |
| 07 | [`07_cherenkov_water/`](07_cherenkov_water/) | Cherenkov in water (air-gap + realistic absorption) on-axis z-fluence | 11 cases (C12/e⁻/proton) |
| 08 | [`08_dispersion_prism/`](08_dispersion_prism/) | SF1 prism wavelength dispersion n(λ) (BEAM kernel) | rainbow spectrum, GPU/CPU |
| 09 | [`09_csi_wls_plate/`](09_csi_wls_plate/) | CsI(Tl) scintillator + WLS plate + proton dose (genstep) | 5-face reflector wrap, fluence z-profile G/C 1.0002 |
| 10 | [`10_analytic_vs_mesh/`](10_analytic_vs_mesh/) | curved surface (sphere/cylinder/torus) analytic vs mesh intersection accuracy | chord = 2R absorption length, analytic ≡ mesh ≡ CPU (sub-0.3%) |

Per-benchmark run scripts, variables, and file layout are documented in each folder's own `README.md`.

## Common (`common/`)

| File | Purpose |
|---|---|
| `topas_env.sh` | resolves the `topas-gpu` binary path (single edit point for every run script) |
| `unit_test_shell.txt` | U1–U5 common TOPAS shell (chained bc408) |
| `bc408.txt` | Buapfcfm/BC408 scintillator material definition |
| `OpticalMaterialSample.txt` | sim_light/torus common material library |
| `simple_biconvex_lens_new.stl` | STL biconvex lens mesh (~7.2 MB) |
| `fig6_design.png` | system-design diagram (used by the 05 torus figure) |

Each benchmark's TOPAS config references these via the relative path `../../common/<file>`.

## Build / requirements

The benchmarks run on the `topas-gpu` binary built at the **repository root** (not here). Build it there:

```bash
cd ..                  # repository root (MetalOpticalPhoton/)
./build_topas_gpu.sh   # auto-detect paths → apply OpenTOPAS patches → build shaders+dylib+TOPAS → install the topas-gpu wrapper
```

Full setup, version rationale, and a manual fallback live in the root docs — this is only a summary:

- **[../INSTALL.md](../INSTALL.md)** — step-by-step setup (toolchain, Geant4, OpenTOPAS source, manual build)
- **[../README.md](../README.md)** — engine overview and quick start
- **[../build_topas_gpu.sh](../build_topas_gpu.sh)** — the build/install script itself
- **[../topas_patches/README.md](../topas_patches/README.md)** — the OpenTOPAS patches applied at build time

Prerequisites (the script verifies but does **not** install them):

- **macOS, Apple Silicon (M series).**
- **Metal toolchain** — shaders compile with `xcrun metal -std=metal3.1`. Command Line Tools alone lack the Metal compiler; on Xcode 26+ it is a separate `xcodebuild -downloadComponent MetalToolchain` (see INSTALL.md §1-1).
- **Homebrew:** `cmake` (≥ 3.20) and `gdcm` — `brew install cmake gdcm`.
- **OpenTOPAS ≥ 4.2.3** — the enforced floor (`TOPAS_REQUIRED="4.2.3"`); the build aborts below it. (Historical note: 4.0.0 additionally collapses energy-dependent `RAYLEIGH` to a wavelength-independent constant, breaking benchmark 03's CPU colour separation.) **Qt6 by default** (required for TOPAS 4.2.3+ — without it the TsQt5 link fails; override `TOPAS_QT=5`).
- **Geant4 ≥ 11.3.2** with the G4DATA datasets (`GEANT4_REQUIRED="11.3.2"`, the build aborts below it); `geant4-config` on `PATH` lets the script auto-locate prefix + data.
- **Python 3** with **numpy, matplotlib, scipy** for the `python3 plot.py` stage.

---

# Appendix — GPU optical engine TOPAS parameter reference

Authoritative reference for the parameters the GPU optical extension reads, verified line-by-line
against the parser source (`topas_extension/TopasParameterParser.cc`, `TsGPUOpticalPhysicsModule.cc`,
`TsGPUOpticalPhysics.cc`) and the engine (`src/MetalOpticalEngine.mm`). It drives the GPU with the
**BEAM kernel** in the optical benchmarks 03·05·06·08; 02 uses the legacy TOPAS-Beam source instead
(it hands the optical-photon primary to the GPU rather than generating on-GPU).

> **Key prefix.** In the tables below each key is written by its **bare name** (e.g. `BeamPhotons`)
> and takes the prefix **`Ph/Default/GPUOptical/`**; the code-block examples spell out the full path,
> ready to paste. Keys shown with their own prefix are the exceptions: `Ts/Seed`, `So/Beam/…`,
> `Su/<Surface>/…`, and `Ge/<Component>/…`.

## 1. Parameter entry paths (three of them)

| Path | Where keys live | How read |
|---|---|---|
| **fPm API** | `Ph/Default/GPUOptical/<Key>`, `Ge/<Comp>/<Key>`, `Su/<Surf>/<Key>`, `Ts/Seed` | TOPAS parameter manager (`Get{Integer,Unitless,Double,String,Boolean}Parameter`) — the `.txt` `i:/u:/d:/s:/b:` prefixes |
| **ParameterFile text parser** | the file named by `s:Ph/Default/GPUOptical/ParameterFile` | `TopasParameterParser` loads that file's material/surface optical properties and process-enable flags and uploads them to the GPU — separate from the fPm `.txt` API |
| **Su/Ge extension keys** | `Su/<Surface>/ForceReflectivity`, `Ge/<Component>/GPUForceMesh` | per-surface / per-volume booleans |

```
s:Ph/Default/GPUOptical/ParameterFile = "../../common/OpticalMaterialSample.txt"  # materials + optical properties
```

## 2. BEAM mode activation

```
sv:Ph/Optical/Modules = 2 "g4em-standard_opt4" "gpuoptical"   # drop g4optical: GPU does generation + transport
i:So/Beam/NumberOfHistoriesInRun = 1          # one G4 trigger event is enough
i:Ph/Default/GPUOptical/BeamPhotons = 5000000 # > 0 turns BEAM kernel mode ON
```

- **Why the BEAM kernel?** The legacy "TOPAS Beam path" builds the optical-photon track on the CPU per event and mutex-pushes each photon to the GPU — under 20-thread MT that race gave non-deterministic agreement (G/C 1.24–1.69 unguarded). The BEAM kernel never builds the track on the CPU: `NumberOfHistoriesInRun = 1` lets G4 process one dummy event while a `thread-per-photon` kernel generates directly → architecturally race-free and **bit-identical at fixed seed**, and fast (1B torus+lens 6.2 s vs CPU 5 h 13 m ≈ 3000×).
- `BeamPhotons > 0` **is** the switch (a present `BeamPhotonsLarge` alone also flips it on). With `g4optical` dropped, `NumberOfHistoriesInRun = 1` suffices; a larger value only adds empty trigger events — the BEAM genstep is consumed once on the first event (re-entry guard in `EndOfEvent`).
- Sections 4–11 are read **only inside** this BEAM block — without `BeamPhotons` (or `BeamPhotonsLarge`) they are ignored.

## 3. Photon count & transport control

| Key | Type | Default | Meaning |
|---|---|---|---|
| `BeamPhotons` | `i:` or `u:` | — (>0 = ON) | photons to generate. Type is auto-detected: `i:` capped at 2³¹−1 ≈ **2.15 B**; **`u:`** (fp64 mantissa, ~9e15) holds 10 B/100 B in one line |
| `EventBatchSize` | `i:` | **5** (events) | G4 **events** accumulated before a GPU dispatch in the *regular (non-BEAM)* path — **not** photons-per-dispatch. Default 5 ≈ GPU-occupancy sweet spot. Results are statistically equivalent across values (per-batch RNG reseeding), not bit-identical — see the race caveat below for large BS in low-transit setups. **Irrelevant in BEAM mode**: the BEAM kernel always splits N into fixed **5 M-photon** GPU chunks (hard cap `MOP_MAX_PHOTONS_PER_BATCH` = 128 M) |
| `MaxStepsPerPhoton` | `i:` | **1 000 000** | max bounces/steps per photon, then the photon is killed. Matches CPU `Ts/MaxStepNumber`. **Lower it** when photons are permanently TIR-trapped (air-gap trapping, infinite-absorption WLS) and would otherwise bounce to 1 M steps and overflow the transit-hit buffer — 05 `torus_diffuse` uses **1000** (~500× faster, identical fluence) |

```
u:Ph/Default/GPUOptical/BeamPhotons = 1.0e10        # 10 B — impossible with i:, one line with u:
i:Ph/Default/GPUOptical/MaxStepsPerPhoton = 1000    # cap trapped photons (05 torus_diffuse)
```

- *Why `EventBatchSize = 5`:* larger values pack more photons per batch, which then re-split into the 5 M chunks → more dispatches and overhead (sim_light 100K: `BS=1000` → 334 s vs `BS=5` → 143 s, 2.3×); BS=5 is optimal at every N. Low-transit-rate setups (05 torus_diffuse, ~0.05% transit) additionally hit a multi-batch race at large BS — exactly why those use the batch-invariant, bit-identical BEAM kernel.
- *Why the 1 M `MaxStepsPerPhoton` default:* an earlier 1000 default killed TIR-trapped WLS photons before re-emission and dropped plate fluence, so it was raised to match CPU TOPAS. 05's air world barely absorbs, so its trapped photons never die by absorption — hence 05 alone lowers both the GPU key and CPU `Ts/MaxStepNumber` back to 1000.
- `BeamPhotonsLarge` is a **deprecated** alias (always read as `u:`; if present it takes precedence over `BeamPhotons`) — still functional, but new configs should use a single `u:BeamPhotons`.

## 4. Source position

| Key | Type·unit | Default | Meaning |
|---|---|---|---|
| `BeamCenterX,Y,Z` | `d:` Length | `0 mm` | beam origin (centre) — `cm`/`nm` etc. also accepted |

## 5. Source direction

| Key | Type | Default | Meaning |
|---|---|---|---|
| `BeamDirX,Y,Z` | `u:` (vector) | `0, -1, 0` | direction of travel, **auto-normalised** (magnitude-independent). Ignored when `BeamAngularDistribution = "Isotropic"`. Unspecified components keep the default `(0, -1, 0)` — `BeamDirY` defaults to −1, so aiming off the −Y axis needs an explicit `u:BeamDirY = 0` (e.g. a +Z beam needs BeamDirY=0, BeamDirZ=1) |

## 6. Source energy (single)

| Key | Type·unit | Default | Meaning |
|---|---|---|---|
| `BeamEnergy` | `d:` Energy | `2.5 eV` | single photon energy (1:1 with wavelength via E=hc/λ; 2.5 eV ≈ 496 nm) |
| `BeamEnergySpread` | `u:` (%) | `0` | **Gaussian sigma** (standard deviation, *not* a half-width): σ = % × E / 100 (same convention as CPU) |

A spectrum (section 7), when present, **takes priority** over `BeamEnergy`/`BeamEnergySpread`.

## 7. Energy spectrum (key to dispersion measurement)

```
s:Ph/Default/GPUOptical/BeamEnergySpectrumType     = "Continuous"   # "Discrete" | "Continuous"
dv:Ph/Default/GPUOptical/BeamEnergySpectrumValues  = 2 1.65 3.10 eV  # energy points
uv:Ph/Default/GPUOptical/BeamEnergySpectrumWeights = 2 1.0 1.0       # weight per point
```

- The presence of `BeamEnergySpectrumType` is what **enables** spectrum handling — declare `Values`/`Weights` without `Type` and the whole spectrum is silently ignored.
- All three of `nValues > 0`, `nWeights == nValues`, `nValues ≤ 256` must hold, or the spectrum is **silently skipped** (256 is the GPU constant-buffer limit; over-256 warns and is ignored).
- **`Discrete`** — emit only at each point; weights are cumulative-summed then **normalised**.
- **`Continuous`** — sub-bin linear interpolation with a trapezoidal CDF; the weight integral (∫w dE) is kept **un-normalised**. Matches CPU TOPAS `TsVGenerator`. (08 = visible 1.65–3.10 eV continuous → prism rainbow.)
- *Why un-normalised:* the shader's inverse-CDF assumes `cumW` is the raw integral (`total = cumW.back()`). An earlier fix removed an unconditional "normalise to [0,1]" that had truncated every continuous spectrum to `E_lo + 1 eV` (dispersion blue end vanished, G/C 1.15 → 1.002); normalisation is now applied to `Discrete` only.

## 8. Position distribution (beam cross-section)

| Key | Type·unit | Default | Meaning |
|---|---|---|---|
| `BeamPosCutoffX,Y` | `d:` Length | `0` | **Flat** rectangular half-width / cutoff |
| `BeamPositionSpreadX,Y` | `d:` Length | `0` | **Gaussian** sigma |
| `BeamPositionCutoffShape` | `s:` | `"Rectangle"` | `"Rectangle"` \| `"Ellipse"` |
| `BeamPositionDistribution` | `s:` | (auto) | `"None"`\|`"Flat"`\|`"Gaussian"` — if unset: spread>0→Gaussian, cutoff>0→Flat, else None |

```
d:Ph/Default/GPUOptical/BeamPosCutoffX = 2.5 mm   # Flat 5×5 mm rectangle
d:Ph/Default/GPUOptical/BeamPosCutoffY = 2.5 mm
```

CPU TOPAS likewise separates `Cutoff` (hard boundary) and `Spread` (Gaussian sigma) — kept identical for validation. (08 uses X 1.0 / Y 0.001 mm → wide-horizontal, thin-vertical so the rainbow spreads in the x–z plane.) Note `"Isotropic"` is **not** meaningful for position — if written it silently behaves as Flat (uniform within the cutoffs).

## 9. Angular distribution (divergence)

| Key | Type·unit | Default | Meaning |
|---|---|---|---|
| `BeamAngSigmaX,Y` | `u:` rad | `0` | **Gaussian** divergence sigma |
| `BeamAngCutoffX,Y` | `u:` rad | `0` | divergence cutoff (Flat) |
| `BeamAngularDistribution` | `s:` | (auto) | `"None"`\|`"Flat"`\|`"Gaussian"`\|**`"Isotropic"`** (4π, ignores `BeamDir`) |

If unset: sigma>0→Gaussian, cutoff>0→Flat, else None (fully collimated). `Isotropic` implements an isotropic point source as a single beam.

## 10. Polarization

```
u:Ph/Default/GPUOptical/BeamPolarizationX = 1   # linear-pol unit vector
u:Ph/Default/GPUOptical/BeamPolarizationY = 0
u:Ph/Default/GPUOptical/BeamPolarizationZ = 0
```

If the vector magnitude is ~0 (default), each photon gets a **random transverse** polarization (the physical default of an unpolarised source); otherwise it is fixed to that linear vector (normalised internally).

## 11. Time distribution

| Key | Type·unit | Default | Meaning |
|---|---|---|---|
| `BeamTimeSpread` | `d:` Time | `0 ns` | emission-time spread, Gaussian sigma (time-resolved measurement) |
| `BeamTimeCutoff` | `d:` Time | `0 ns` | emission-time cutoff (`0` = no clip) |

Default 0 = all photons emitted at t=0.

## 12. Multi-beam (up to 32 sources)

Add sources with the `Beam2`…`Beam32` prefix; each holds its own `BeamCenter/Dir/Energy/PosCutoff/AngSigma/AngCutoff` (and `BeamPhotons`). 32 = the module's `Beam2`…`Beam32` key-scan bound (not an engine buffer limit).

```
i:Ph/Default/GPUOptical/Beam2/BeamPhotons = 1000000
d:Ph/Default/GPUOptical/Beam2/BeamCenterX = 5 mm
u:Ph/Default/GPUOptical/Beam2/BeamDirY    = 0   # BeamDirY defaults to -1 — zero it explicitly
u:Ph/Default/GPUOptical/Beam2/BeamDirZ    = 1
```

- **Global-only (shared from Beam1):** distribution modes and `Shape`, `Polarization`, `Spectrum`, `BeamPositionSpread`, `BeamEnergySpread`, `BeamTime*`. Extra beams carry only the per-beam keys listed above.
- *Caveat:* a per-beam `BeamPosCutoff`/`BeamAngSigma` is **inert** unless the global (Beam1) distribution mode resolves to Flat/Gaussian — the mode is global, so set the envelope on Beam1.

## 13. GPU/CPU validation mode

| Key | Type | Default | Meaning |
|---|---|---|---|
| `ValidationMode` | `b:` | `false` | when `true`, optical photons are **not killed**, so CPU `g4optical` and the GPU both track them → direct GPU transit-fluence vs CPU `Fluence` comparison on the same primaries |

## 14. Reproducibility

| Key | Type | Default | Meaning |
|---|---|---|---|
| `Ts/Seed` | `i:` | `1` | global TOPAS seed, **propagated to the GPU RNG base** (`SetBaseSeed`) so BEAM-mode runs differ per seed — essential for multi-seed averaging. (Always propagated: with no `Ts/Seed` the GPU gets 1, not the engine's internal 42.) |

## 15. Surface / geometry extensions

| Key | Type | Default | Meaning |
|---|---|---|---|
| `Su/<Surface>/ForceReflectivity` | `b:` | `false` | use the surface's REFLECTIVITY value as the **absolute** reflection probability (reflect with prob R, else Snell-refract), bypassing the Geant4 Fresnel gate; queried on both skin and border surface paths |
| `Ge/<Component>/GPUForceMesh` | `b:` | `false` | skip the solid's analytic registration and use the mesh (BVH) path (OR-ed with env `MOP_SPHERE_ANALYTIC`/`MOP_CYLINDER_ANALYTIC`); the only mesh toggle for a torus — drives benchmark 10 (analytic-vs-mesh) |

## 16. Logging

| Key | Type | Default | Meaning |
|---|---|---|---|
| `LogLevel` | `i:` | **1** | `0` = minimal (a few one-shot banners — seed, BEAM mode, final stats — always print); `1` = geometry register + end-of-run summary; `≥2` = per-event diagnostics + per-chunk verbose. (Configs that write `0` are choosing an example value; the **code default is 1**.) |

*Why per-chunk logs need ≥2:* a 1 B BEAM run is ~200 chunks, so the per-batch flush log sits at `≥2` to keep the default output readable.
