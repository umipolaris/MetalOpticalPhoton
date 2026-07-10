#!/usr/bin/env python3
"""U2v5 batch — new geometry (BC408 box, source IN BC408, +x face = boundary at x=0).

Geometry (per user 2026-04-25 option A):
  World      Air, 260 × 150 × 150 mm (HLX=130, HLY=HLZ=75) — extended in x for BC408
  BC408      Buapfcfm n=1.58, 120 mm cube (HLX=HLY=HLZ=60), TransX=-60
             → x=-120..0, ±60 in y,z. +x face at x=0.
  AirDet     Air thin slab on +x side at x=+0.5..+1.5 (HLX=0.5, TransX=+1)
  Source     IN BC408 on radius-50 circle around (0,0,0):
             position (-50 cos θ, 0, +50 sin θ)  — all 5°-85° within BC408
             beam direction (cos θ, 0, -sin θ) toward (0,0,0) (BC408 +x face)
             RotY = 90 + θ
  Angles     5°-85° (all fit)

Scorers:
  T          going_in @ AirDet/XMinusSurface (catches refracted T at x=+0.5)
  R          inferred = N_source - T (TIR for θ>39.27° → photon trapped/internal absorb,
             T=0 → R=1 by conservation)
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import time
from pathlib import Path

BUILT = Path(__file__).resolve().parent.parent / "results" / "built"
ANGLES = [int(a) for a in os.environ.get("ANGLES", "5 15 25 30 35 38 40 45 50 60 70 85").replace(",", " ").split()]
TOPAS = os.environ.get("TOPAS") or shutil.which("topas-gpu") or "/Applications/TOPAS/OpenTOPAS-install-gpu/bin/topas-gpu"
MODES = os.environ.get("MODE", "gpu cpu").replace(",", " ").split()
THREADS = os.environ.get("THREADS", "").strip()  # when set, inject i:Ts/NumberOfThreads


def src_pos(theta_deg):
    th = math.radians(theta_deg)
    return -50.0 * math.cos(th), 0.0, +50.0 * math.sin(th)


def roty(theta_deg):
    # TOPAS RotY(α): (0,0,1) → (-sin α, 0, cos α). For (cos θ, 0, -sin θ) need α=-(90+θ).
    return -(90.0 + theta_deg)


TEMPLATE_GPU = """\
includeFile = ../../common/unit_test_shell.txt
sv:Ph/Optical/Modules = 3 "g4em-standard_opt4" "g4optical" "gpuoptical"
i:Ph/Default/GPUOptical/EventBatchSize = 1000
i:Ts/Seed = 42

# World 260 × 150 × 150 Air (extended in x for BC408)
d:Ge/World/HLX = 130 mm
d:Ge/World/HLY = 75 mm
d:Ge/World/HLZ = 75 mm
s:Ge/World/Material = "Air"

# BC408 (Buapfcfm n=1.58)
sv:Ma/Buapfcfm/Components = 1 "Carbon"
uv:Ma/Buapfcfm/Fractions = 1 1.0
d:Ma/Buapfcfm/Density = 1.032 g/cm3
b:Ma/Buapfcfm/EnableOpticalProperties = "True"
dv:Ma/Buapfcfm/RIndex/Energies = 2 1.5 5.0 eV
uv:Ma/Buapfcfm/RIndex/Values = 2 1.58 1.58
dv:Ma/Buapfcfm/AbsLength/Energies = 2 1.5 5.0 eV
dv:Ma/Buapfcfm/AbsLength/Values = 2 100000 100000 cm

# BC408: 120 mm cube, +x face at x=0
s:Ge/Sci/Parent = "World"
s:Ge/Sci/Type = "TsBox"
s:Ge/Sci/Material = "Buapfcfm"
d:Ge/Sci/HLX = 60 mm
d:Ge/Sci/HLY = 60 mm
d:Ge/Sci/HLZ = 60 mm
d:Ge/Sci/TransX = -60 mm


# BC408_Black absorber material (same n=1.58, AbsLen=1mm — Fresnel skip + fast absorb)
sv:Ma/BuapfcfmBlack/Components = 1 "Carbon"
uv:Ma/BuapfcfmBlack/Fractions = 1 1.0
d:Ma/BuapfcfmBlack/Density = 1.032 g/cm3
b:Ma/BuapfcfmBlack/EnableOpticalProperties = "True"
dv:Ma/BuapfcfmBlack/RIndex/Energies = 2 1.5 5.0 eV
uv:Ma/BuapfcfmBlack/RIndex/Values = 2 1.58 1.58
dv:Ma/BuapfcfmBlack/AbsLength/Energies = 2 1.5 5.0 eV
dv:Ma/BuapfcfmBlack/AbsLength/Values = 2 1.0 1.0 mm

# 5-face absorber wraps around BC408 (Sci): -x, +y, -y, +z, -z
# +x face left open for AirDet measurement
s:Ge/Wrap_Xm/Parent = "World"
s:Ge/Wrap_Xm/Type = "TsBox"
s:Ge/Wrap_Xm/Material = "BuapfcfmBlack"
d:Ge/Wrap_Xm/HLX = 0.5 mm
d:Ge/Wrap_Xm/HLY = 60 mm
d:Ge/Wrap_Xm/HLZ = 60 mm
d:Ge/Wrap_Xm/TransX = -120.5 mm

s:Ge/Wrap_Yp/Parent = "World"
s:Ge/Wrap_Yp/Type = "TsBox"
s:Ge/Wrap_Yp/Material = "BuapfcfmBlack"
d:Ge/Wrap_Yp/HLX = 60 mm
d:Ge/Wrap_Yp/HLY = 0.5 mm
d:Ge/Wrap_Yp/HLZ = 60 mm
d:Ge/Wrap_Yp/TransX = -60 mm
d:Ge/Wrap_Yp/TransY = 60.5 mm

s:Ge/Wrap_Ym/Parent = "World"
s:Ge/Wrap_Ym/Type = "TsBox"
s:Ge/Wrap_Ym/Material = "BuapfcfmBlack"
d:Ge/Wrap_Ym/HLX = 60 mm
d:Ge/Wrap_Ym/HLY = 0.5 mm
d:Ge/Wrap_Ym/HLZ = 60 mm
d:Ge/Wrap_Ym/TransX = -60 mm
d:Ge/Wrap_Ym/TransY = -60.5 mm

s:Ge/Wrap_Zp/Parent = "World"
s:Ge/Wrap_Zp/Type = "TsBox"
s:Ge/Wrap_Zp/Material = "BuapfcfmBlack"
d:Ge/Wrap_Zp/HLX = 60 mm
d:Ge/Wrap_Zp/HLY = 60 mm
d:Ge/Wrap_Zp/HLZ = 0.5 mm
d:Ge/Wrap_Zp/TransX = -60 mm
d:Ge/Wrap_Zp/TransZ = 60.5 mm

s:Ge/Wrap_Zm/Parent = "World"
s:Ge/Wrap_Zm/Type = "TsBox"
s:Ge/Wrap_Zm/Material = "BuapfcfmBlack"
d:Ge/Wrap_Zm/HLX = 60 mm
d:Ge/Wrap_Zm/HLY = 60 mm
d:Ge/Wrap_Zm/HLZ = 0.5 mm
d:Ge/Wrap_Zm/TransX = -60 mm
d:Ge/Wrap_Zm/TransZ = -60.5 mm

# AirDet: thin Air slab on +x side of BC408 boundary
s:Ge/AirDet/Parent = "World"
s:Ge/AirDet/Type = "TsBox"
s:Ge/AirDet/Material = "Air"
d:Ge/AirDet/HLX = 0.5 mm
d:Ge/AirDet/HLY = 60 mm
d:Ge/AirDet/HLZ = 60 mm
d:Ge/AirDet/TransX = 1 mm

s:So/Beam/Type = "Beam"
s:So/Beam/Component = "BeamPos"
s:So/Beam/BeamParticle = "opticalphoton"
d:So/Beam/BeamEnergy = 2.5 eV
u:So/Beam/BeamEnergySpread = 0.0
s:So/Beam/BeamPositionDistribution = "Flat"
s:So/Beam/BeamPositionCutoffShape = "Rectangle"
d:So/Beam/BeamPositionSpreadX = 0.0 mm
d:So/Beam/BeamPositionSpreadY = 0.0 mm
d:So/Beam/BeamPositionCutoffX = 0.0001 mm
d:So/Beam/BeamPositionCutoffY = 0.0001 mm
s:So/Beam/BeamAngularDistribution = "Flat"
d:So/Beam/BeamAngularSpreadX = 0.0 deg
d:So/Beam/BeamAngularSpreadY = 0.0 deg
d:So/Beam/BeamAngularCutoffX = 0.0001 deg
d:So/Beam/BeamAngularCutoffY = 0.0001 deg
u:So/Beam/BeamPolarizationX = 0.0
u:So/Beam/BeamPolarizationY = 1.0
u:So/Beam/BeamPolarizationZ = 0.0
i:So/Beam/NumberOfHistoriesInRun = 100000

# Source: IN BC408, radius-50 circle around (0,0,0). Beam → (0,0,0) at incidence θ.
s:Ge/BeamPos/Parent = "World"
s:Ge/BeamPos/Type = "Group"
d:Ge/BeamPos/TransX = {src_x:.6f} mm
d:Ge/BeamPos/TransY = 0 mm
d:Ge/BeamPos/TransZ = {src_z:.6f} mm
d:Ge/BeamPos/RotY = {roty:.6f} deg

# T (transmitted): SF1 going_in @ AirDet/XMinusSurface
s:Sc/U2v5_T/Quantity = "GPUOpticalPhotonSurfaceTrackCount"
s:Sc/U2v5_T/Component = "AirDet"
s:Sc/U2v5_T/Surface = "AirDet/XMinusSurface"
s:Sc/U2v5_T/OnlyIncludeParticlesGoing = "in"
b:Sc/U2v5_T/OutputToConsole = "False"
s:Sc/U2v5_T/OutputType = "csv"
s:Sc/U2v5_T/OutputFile = "U2v5_gpu_th{th}_T_sf1"
s:Sc/U2v5_T/IfOutputFileAlreadyExists = "Overwrite"
"""

TEMPLATE_CPU = TEMPLATE_GPU.replace(
    'sv:Ph/Optical/Modules = 3 "g4em-standard_opt4" "g4optical" "gpuoptical"',
    'sv:Ph/Optical/Modules = 2 "g4em-standard_opt4" "g4optical"'
).replace(
    'U2v5_gpu_th', 'U2v5_cpu_th'
).replace(
    'Quantity = "GPUOpticalPhotonSurfaceTrackCount"\ns:Sc/U2v5_T/Component = "AirDet"',
    'Quantity = "SurfaceTrackCount"\nsv:Sc/U2v5_T/OnlyIncludeParticlesNamed = 1 "opticalphoton"\ns:Sc/U2v5_T/Component = "AirDet"'
)


def build_one(mode: str, th: int) -> Path:
    template = TEMPLATE_GPU if mode == "gpu" else TEMPLATE_CPU
    sx, sy, sz = src_pos(th)
    text = template.format(src_x=sx, src_z=sz, roty=roty(th), th=th)
    path = BUILT / f"U2v5_{mode}_th{th}.txt"
    if THREADS:
        text = text.replace("i:Ts/Seed = 42", f"i:Ts/Seed = 42\ni:Ts/NumberOfThreads = {THREADS}", 1)
    path.write_text(text)
    return path


def run_one(input_path: Path):
    t0 = time.time()
    res = subprocess.run(
        [TOPAS, input_path.name],
        cwd=str(BUILT),
        capture_output=True, text=True, timeout=600,
    )
    dt = time.time() - t0
    return dt, res.returncode


def main():
    print(f"Generating + running {len(ANGLES)*len(MODES)} U2v5 jobs (modes={MODES}) ...")
    for i, th in enumerate(ANGLES):
        for mode in MODES:
            inp = build_one(mode, th)
            dt, rc = run_one(inp)
            print(f"  [{i+1:2d}/{len(ANGLES)}] {mode} th={th:2d}  {dt:5.1f}s rc={rc}")


if __name__ == "__main__":
    main()
