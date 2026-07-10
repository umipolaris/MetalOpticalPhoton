#!/usr/bin/env python3
"""U1v5 batch — new geometry (Air/Glass at x=0 vertical interface).

Geometry (per user 2026-04-25):
  World      Air, 150 mm cube (HLX=HLY=HLZ=75)
  Glass      n=1.5, x=0..+75 (HLX=37.5, TransX=+37.5)
  AirRefl    Air thin slab x=-1.5..-0.5 (HLX=0.5, TransX=-1)
  Source     on radius-50 circle around (0,0,0):
             position (-50 cos θ, 0, +50 sin θ)
             beam direction (cos θ, 0, -sin θ) toward (0,0,0)
             RotY = 90 + θ (TOPAS default beam dir +Z)
  Angles     5°-85° (all fit inside World)

Scorers (standard SF1 measurement):
  T          going_in @ Glass/XMinusSurface (catches refracted T)
  R          going_in @ AirRefl/XPlusSurface (catches reflected R)
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

# World 150 mm cube Air
d:Ge/World/HLX = 75 mm
d:Ge/World/HLY = 75 mm
d:Ge/World/HLZ = 75 mm
s:Ge/World/Material = "Air"

# Glass material (n=1.5, AbsLen=1cm)
sv:Ma/GlassN15Abs/Components = 1 "Silicon"
uv:Ma/GlassN15Abs/Fractions = 1 1.0
d:Ma/GlassN15Abs/Density = 2.2 g/cm3
b:Ma/GlassN15Abs/EnableOpticalProperties = "True"
dv:Ma/GlassN15Abs/RIndex/Energies = 2 1.5 5.0 eV
uv:Ma/GlassN15Abs/RIndex/Values = 2 1.5 1.5
dv:Ma/GlassN15Abs/AbsLength/Energies = 2 1.5 5.0 eV
dv:Ma/GlassN15Abs/AbsLength/Values = 2 1.0 1.0 cm

# Glass: vertical interface at x=0, Glass on +x half (HLY/HLZ slightly < World to avoid coincident face)
s:Ge/Glass/Parent = "World"
s:Ge/Glass/Type = "TsBox"
s:Ge/Glass/Material = "GlassN15Abs"
d:Ge/Glass/HLX = 37 mm
d:Ge/Glass/HLY = 70 mm
d:Ge/Glass/HLZ = 70 mm
d:Ge/Glass/TransX = 37 mm

# AirRefl: thin Air slab in Air region, just left of interface
s:Ge/AirRefl/Parent = "World"
s:Ge/AirRefl/Type = "TsBox"
s:Ge/AirRefl/Material = "Air"
d:Ge/AirRefl/HLX = 0.5 mm
d:Ge/AirRefl/HLY = 70 mm
d:Ge/AirRefl/HLZ = 70 mm
d:Ge/AirRefl/TransX = -1 mm

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
u:So/Beam/BeamPolarizationX = 1.0
u:So/Beam/BeamPolarizationY = 0.0
u:So/Beam/BeamPolarizationZ = 0.0
i:So/Beam/NumberOfHistoriesInRun = 100000

# Source: radius-50 circle around (0,0,0). Beam → (0,0,0) at incidence θ.
s:Ge/BeamPos/Parent = "World"
s:Ge/BeamPos/Type = "Group"
d:Ge/BeamPos/TransX = {src_x:.6f} mm
d:Ge/BeamPos/TransY = 0 mm
d:Ge/BeamPos/TransZ = {src_z:.6f} mm
d:Ge/BeamPos/RotY = {roty:.6f} deg

# T (transmitted): SF1 going_in @ Glass/XMinusSurface (entering Glass at x=0)
s:Sc/U1v5_T/Quantity = "GPUOpticalPhotonSurfaceTrackCount"
s:Sc/U1v5_T/Component = "Glass"
s:Sc/U1v5_T/Surface = "Glass/XMinusSurface"
s:Sc/U1v5_T/OnlyIncludeParticlesGoing = "in"
b:Sc/U1v5_T/OutputToConsole = "False"
s:Sc/U1v5_T/OutputType = "csv"
s:Sc/U1v5_T/OutputFile = "U1v5_ppol_gpu_th{th}_T_sf1"
s:Sc/U1v5_T/IfOutputFileAlreadyExists = "Overwrite"

# R (reflected): SF1 going_in @ AirRefl/XPlusSurface (caught by AirRefl on Air side)
# Incident beam crosses AirRefl going +x → going_OUT at XPlusSurface (filter excludes)
# R photon crosses AirRefl going -x → going_IN at XPlusSurface (counted)
s:Sc/U1v5_R/Quantity = "GPUOpticalPhotonSurfaceTrackCount"
s:Sc/U1v5_R/Component = "AirRefl"
s:Sc/U1v5_R/Surface = "AirRefl/XPlusSurface"
s:Sc/U1v5_R/OnlyIncludeParticlesGoing = "in"
b:Sc/U1v5_R/OutputToConsole = "False"
s:Sc/U1v5_R/OutputType = "csv"
s:Sc/U1v5_R/OutputFile = "U1v5_ppol_gpu_th{th}_R_sf1"
s:Sc/U1v5_R/IfOutputFileAlreadyExists = "Overwrite"
"""

TEMPLATE_CPU = TEMPLATE_GPU.replace(
    'sv:Ph/Optical/Modules = 3 "g4em-standard_opt4" "g4optical" "gpuoptical"',
    'sv:Ph/Optical/Modules = 2 "g4em-standard_opt4" "g4optical"'
).replace(
    'U1v5_ppol_gpu_th', 'U1v5_ppol_cpu_th'
).replace(
    'Quantity = "GPUOpticalPhotonSurfaceTrackCount"\ns:Sc/U1v5_T/Component = "Glass"',
    'Quantity = "SurfaceTrackCount"\nsv:Sc/U1v5_T/OnlyIncludeParticlesNamed = 1 "opticalphoton"\ns:Sc/U1v5_T/Component = "Glass"'
).replace(
    'Quantity = "GPUOpticalPhotonSurfaceTrackCount"\ns:Sc/U1v5_R/Component = "AirRefl"',
    'Quantity = "SurfaceTrackCount"\nsv:Sc/U1v5_R/OnlyIncludeParticlesNamed = 1 "opticalphoton"\ns:Sc/U1v5_R/Component = "AirRefl"'
)


def build_one(mode: str, th: int) -> Path:
    template = TEMPLATE_GPU if mode == "gpu" else TEMPLATE_CPU
    sx, sy, sz = src_pos(th)
    text = template.format(src_x=sx, src_z=sz, roty=roty(th), th=th)
    path = BUILT / f"U1v5_{mode}_th{th}.txt"
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
    print(f"Generating + running {len(ANGLES)*len(MODES)} U1v5_ppol jobs (modes={MODES}) ...")
    for i, th in enumerate(ANGLES):
        for mode in MODES:
            inp = build_one(mode, th)
            dt, rc = run_one(inp)
            print(f"  [{i+1:2d}/{len(ANGLES)}] {mode} th={th:2d}  {dt:5.1f}s rc={rc}")


if __name__ == "__main__":
    main()
