# 01_speed — sim_light execution speed and agreement

This benchmark runs `sim_light` (proton → plastic scintillator → 45° mirror + BK7 lens → sensor)
on the GPU (Metal) and Geant4 CPU with the same number of histories to measure **execution time**
and **fluence agreement**. It supports two scoring resolutions: standard (1×1×50) and high-resolution (816×624×50).
This benchmark has no intro figure; it contributes the speed/accuracy numbers in the paper's
**Table 3** (detector simulation timing and accuracy).

## Setup

A 100 MeV proton enters the plastic scintillator → scintillation photons → reflection off the 45° mirror →
imaging through the BK7 biconvex lens (STL mesh) → depth-resolved fluence integration at `ImagingSensor`.

### Geometry / materials

| Volume | Type | Material | Dimensions (half-length) / notes |
|---|---|---|---|
| `World` | — | Air | HLX 20 cm, HLY 10 cm |
| `Plastic` | TsBox | `Buapfcfm` (plastic scintillator) | 25 × 50 × 25 mm, Birks `kB`=0.126 mm/MeV |
| `Mirror` | TsBox | `Borosilicate` | 33 × 50 × 0.5 mm, RotY=45°, `surfaceMirror` `R`=0.9 (`dielectric_metal`, polished, unified) |
| `MySTL` | TsCAD | `BK7` | `simple_biconvex_lens_new` STL (`../../common/` symbolic link), RotY=180° |
| `ImagingSensor` | TsBox | Air | 3.672 × 2.808 × 5.0 mm, `TransZ`=imgTransZ − `imgShiftToZ` (5.0 mm, focal plane) |

### Source / Beam

| Item | Value |
|---|---|
| Particle / energy | proton, 100 MeV (`BeamEnergySpread`=0) |
| Position distribution | Gaussian, `Ellipse` cutoff 25 mm (5σ), `AngularDistribution`=None |
| Number of histories | 1e3 / 1e4 / 1e5 (`SIZES` 1K / 10K / 100K) |
| GPU photon generation | scintillation genstep → GPU propagate (`gpuoptical` module) |
| CPU photon generation | Geant4 scintillation (`g4optical` module) |

### Scorer

| Item | GPU | CPU |
|---|---|---|
| Quantity | `GPUOpticalPhotonFluence` | `Fluence` (opticalphoton filter) |
| Component | `ImagingSensor` | `ImagingSensor` |
| binning | XBins=YBins=1, ZBins=50 (standard) / 816×624×50 (high-resolution, 9 µm pixel, 25.4M voxel) | same |
| Output | `results/SensorDepth_GPU.csv` | `results/SensorDepth_CPU.csv` |

> The GPU/CPU configs differ only in the optical module (`gpuoptical` ↔ `g4optical`), the scorer Quantity (plus the CPU-side opticalphoton filter), the default thread count (20 vs 30; the run scripts override it), a GPU-only `LogLevel`,
> and the output filename; the geometry, materials, and light source are identical.

## Running

| Script | Device | binning | Output | Default thread |
|---|---|---|---|---|
| `run_gpu_standard.sh` | GPU (Metal) | 1×1×50 | `results/gpu_std_<tag>.log` | T=1 |
| `run_gpu_hires.sh`    | GPU (Metal) | 816×624×50 | `results/gpu_hi_<tag>.log` | T=1 |
| `run_cpu_standard.sh` | CPU (Geant4) | 1×1×50 | `results/cpu_std_<tag>.log` | T=30 |
| `run_cpu_hires.sh`    | CPU (Geant4) | 816×624×50 | `results/cpu_hi_<tag>.log` | T=30 |

Each script runs `sim_light` automatically with the specified number of histories and prints the TOPAS
`Execution: Real` time (+ fluence sum). The original config is preserved; thread/binning are injected into a
temporary config with `sed`, which is then run and deleted (for high-resolution, `Sc/SensorPhoton/{XBins,YBins}`=816/624
are also injected).

```bash
cd benchmark/01_speed

./run_gpu_standard.sh                    # default 1K
./run_gpu_standard.sh "10K 100K"         # specify sizes as argument
SIZES="1K 10K 100K" ./run_cpu_hires.sh   # full scan via env
THREADS=30 ./run_gpu_standard.sh "100K"  # GPU thread override (same result and speed)
```

The size resolution priority is `SIZES="${1:-${SIZES:-1K}}"` — **argument → env `SIZES` → default 1K**.
Available sizes are 1K / 10K / 100K, for which `configs/sim_light_{gpu,cpu}_{1K,10K,100K}.txt` exist.

### Environment variables

| Variable | Default | Scope | Description |
|---|---|---|---|
| `TOPAS` | `topas-gpu` (`../common/topas_env.sh`) | all | TOPAS binary path (**patched required**) |
| `SIZES` | `1K` | all | List of history counts (can also be given as the first argument) |
| `THREADS` | GPU 1 / CPU 30 | all | `i:Ts/NumberOfThreads` injected value |

## Notes

- **thread policy**: On GPU it is propagate-bound, so T=1 = T=30 (same speed) and T=1 is bit-reproducible →
  **recommended T=1**. On CPU, T=30 (full use of the 30 Apple Silicon cores) is faster.
- **`G4TRACE_DIR=OFF` applied automatically**: a runtime switch that **disables the disk logging** of the
  `g4trace` instrumentation compiled into the binary. If not disabled, G4 writes `/tmp/g4trace*.log` at every
  photon boundary, making the CPU tens to hundreds of times slower (e.g. 1K CPU 7400 s → 54 s, 137×).
  **The physics result is bit-identical whether on or off** (only the instrumentation logging is turned off).
- **patched binary required**: unpatched `topas` overshoots the optical fluence by 4×.
- **CSV overwrite / automatic cleanup**: each run overwrites `results/SensorDepth_{GPU,CPU}.csv` (the next
  run reuses/overwrites it). The high-resolution (25.4M voxel) CSV is ~700 MB, so it is deleted right after the sum is extracted.
- **Output meaning**: `Execution(s)` = TOPAS `Execution: Real` (pure simulation time, excluding init/finalize),
  `sum(/mm2)` = sum of the fluence scorer over all voxels (for comparing the GPU/CPU sum ratio).

## File structure

```
01_speed/
├── run_gpu_standard.sh        # GPU standard 1×1×50 (T=1)
├── run_gpu_hires.sh           # GPU high-resolution 816×624×50
├── run_cpu_standard.sh        # CPU standard 1×1×50 (T=30)
├── run_cpu_hires.sh           # CPU high-resolution 816×624×50
├── configs/
│   ├── sim_light_gpu_{1K,10K,100K}.txt   # GPU(gpuoptical), GPUOpticalPhotonFluence
│   ├── sim_light_cpu_{1K,10K,100K}.txt   # CPU(g4optical), Fluence(opticalphoton)
│   └── simple_biconvex_lens_new.stl      # → ../../common/… (symlink created by the run scripts; gitignored)
└── results/                   # CSV, log, and png are generated when scripts run, gitignored (only .gitkeep tracked)
    ├── {gpu,cpu}_{std,hi}_<tag>.log      # Execution Real log
    └── SensorDepth_{GPU,CPU}.csv         # depth-resolved fluence (overwritten each run)
```

Master index: [`../README.md`](../README.md)
