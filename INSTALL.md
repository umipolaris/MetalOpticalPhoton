# Installation Guide (macOS Apple Silicon)

MetalOpticalPhoton builds an Apple-Metal GPU optical-photon engine into **OpenTOPAS (Geant4)**.
This document is the **zero-to-run** sequence on a fresh Mac.

## 0. Summary

```bash
# (install the prerequisites in section 1 first)
git clone https://github.com/umipolaris/MetalOpticalPhoton.git
cd MetalOpticalPhoton
./build_topas_gpu.sh            # mode prompt → f (full build)
topas-gpu your_sim.txt
```

`build_topas_gpu.sh` does everything in one pass: **path auto-detection → OpenTOPAS patch
auto-apply → build → install → run-wrapper generation**. If your paths are non-standard, point the
script at them with environment variables (§2).

## 1. Prerequisites (install yourself)

The build script only **detects and validates** the following — it does not install them.
Have them ready beforehand.

### 1-1. Toolchain
- macOS on **Apple Silicon** (M series)
- **Xcode + the Metal toolchain** — needed for shader compilation (`xcrun metal`).
  > ⚠ **Command Line Tools (`xcode-select --install`) do NOT include the Metal compiler.** From
  > Xcode 26+ the Metal toolchain is a separate download from Xcode itself. If the build script
  > prints `✗ Metal 컴파일러 없음 — Xcode + Metal 툴체인 필요 …` ("Metal compiler missing —
  > Xcode + Metal toolchain required"), follow the steps below.
  ```bash
  # 1. Install Xcode from the App Store, then switch the active developer directory to Xcode
  sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
  sudo xcodebuild -license accept

  # 2. Install first-launch components (needed if you see the IDESimulatorFoundation plugin error)
  xcodebuild -runFirstLaunch

  # 3. Download the Metal toolchain (~700 MB)
  xcodebuild -downloadComponent MetalToolchain

  # 4. Verify — success if it prints "Apple metal version ..."
  xcrun metal --version
  ```
  > **Note:** the Metal toolchain is mounted as a cryptex and **can unmount after a reboot.** If
  > `xcrun metal --version` fails again after rebooting, remount it with the same command (no
  > re-download): `xcodebuild -downloadComponent MetalToolchain`
- **Homebrew** → `cmake`, `gdcm`
  ```bash
  brew install cmake gdcm
  ```

### 1-2. Geant4
- Install **Geant4 11.3.2 or later** + download the datasets (G4DATA), following the official
  Geant4 instructions.
- If `geant4-config` is on PATH, the build script finds the prefix and data paths automatically,
  and it aborts the build below `11.3.2`.

### 1-3. OpenTOPAS source
- upstream: <https://github.com/OpenTOPAS/OpenTOPAS>
- **Supported version: `v4.2.3` or later.** The build script **aborts** below `v4.2.3`
  (CMakeLists version check). The vanilla source patch is applied automatically by the build
  script — see [`topas_patches/README.md`](topas_patches/README.md) for details.
- For OpenTOPAS's **own build prerequisites (Qt6 etc.)**, follow the official OpenTOPAS
  documentation. You only need the source checked out — the actual build is done by our script.
  ```bash
  git clone https://github.com/OpenTOPAS/OpenTOPAS.git
  cd OpenTOPAS && git checkout v4.2.3      # 4.2.3 or later
  ```

## 2. Build & install

```bash
git clone <MetalOpticalPhoton repo>
cd MetalOpticalPhoton
./build_topas_gpu.sh                        # choose f (full) at the prompt
```

What the script does:
1. **Auto-detects** the Geant4 / OpenTOPAS source / GDCM / G4DATA paths (and tells you which env
   var to set for anything it cannot find)
2. **Auto-applies the bug patch** to vanilla OpenTOPAS — a single patch file
   (`opentopas_local.patch`, touching 4 source files = 2 logical fixes)
   (idempotent; version warning + dry-run + 4/4 file verification)
3. Builds the Metal shaders → `metallib`, and the GPU engine `dylib`
4. Builds OpenTOPAS + the GPU extension
5. Installs + generates the `topas-gpu` run wrapper (baking in the Geant4/G4DATA paths)

### Non-standard paths — environment-variable overrides

```bash
Geant4_DIR=/my/geant4 TOPAS_SRC=/my/OpenTOPAS GDCM_DIR=/my/gdcm/lib/gdcm-3.x \
  ./build_topas_gpu.sh full
```

| Variable | Meaning | Default |
|---|---|---|
| `Geant4_DIR` | Geant4 install prefix | `geant4-config` / common locations |
| `TOPAS_SRC` | OpenTOPAS source tree | common locations |
| `TOPAS_INSTALL` / `TOPAS_BUILD` | install/build output | `${TOPAS_SRC}-install-gpu` / `-build-gpu` |
| `GDCM_DIR` | GDCM versioned lib directory | `brew --prefix gdcm` / common locations |
| `TOPAS_G4_DATA_DIR` | Geant4 data (G4DATA) | the selected Geant4's bundled `share/Geant4*/data` · common locations, `geant4-config --datasets` as last resort |
| `JOBS` | parallel compile jobs (`make -j`) | all cores |
| `START` | start stage `2`/`3`/`4`/`5` (resume from a failed stage) | `2` |
| `SKIP_CMAKE=1` | in Step 4, skip the cmake re-configure and run `make` only | (unset) |
| `NO_COLOR=1` | disable terminal colors | (color on if TTY) |

> **If auto-detection fails**: on an interactive terminal the script **asks you to type the path
> directly**; non-interactive (pipe/CI) runs get an error telling you which env var to set. In
> other words, everything findable is automatic — you only fill in what could not be found.

### When a compile fails — resume from that stage

Compiles fail from time to time (header clashes, Qt/Geant4 version issues, …). You don't have to
start over — resume **from the failed stage**; on failure the script prints the exact retry command.

```bash
# stages: 2=shaders  3=engine dylib  4=TOPAS build  5=install
START=4 ./build_topas_gpu.sh full              # from Step 4 (TOPAS build), reusing 2·3 artifacts
START=4 SKIP_CMAKE=1 ./build_topas_gpu.sh full # continue with make only (skip cmake re-configure)
JOBS=4 ./build_topas_gpu.sh full               # compile with 4 jobs (low memory / debugging)
```

> The full `cmake`/`make` logs of Step 4 are kept at `${TOPAS_BUILD}/cmake.log` · `make.log` —
> check there for the cause of a failure.

## 3. Run

```bash
topas-gpu your_sim.txt
```

Add the GPU optical module to your TOPAS input (gpuoptical only, without g4optical):
```
s:Ph/ListName = "Optical"
sv:Ph/Optical/Modules = 2 "g4em-standard_opt4" "gpuoptical"
```

For validation, see benchmarks 01–10 under `benchmark/` (each folder has a README).

## 4. Rebuild the engine only (no TOPAS rebuild)

If you only changed the engine/shader code, use the fast **wrapper mode**:
```bash
./build_topas_gpu.sh wrapper
```
It rebuilds only the `dylib`/`metallib`, swaps them into the existing install, and refreshes the
wrapper (no TOPAS rebuild, no patching).
※ If you changed the TOPAS **extension C++** (scorers / physics module), rebuild with `full`.

## 5. OpenTOPAS patches

The patches applied automatically at build time (why they are needed and what they fix):
[`topas_patches/README.md`](topas_patches/README.md),
[`topas_patches_summary.md`](topas_patches_summary.md).

## 6. Manual install (when the build script doesn't work — the classic way)

`build_topas_gpu.sh` may fail on unusual environments. Below are the **exact commands** the script
runs internally, to execute yourself. First fill in the paths for your machine:

```bash
export MOP="$(pwd)"                                  # MetalOpticalPhoton clone location
export TOPAS_SRC=/path/to/OpenTOPAS                  # OpenTOPAS source (build/install derived below)
export TOPAS_BUILD="${TOPAS_SRC}-build-gpu"
export TOPAS_INSTALL="${TOPAS_SRC}-install-gpu"
export GEANT4_DIR=/path/to/geant4-install            # Geant4 install (cmake config location)
export GDCM_DIR="$(brew --prefix gdcm)/lib/gdcm-3.2" # dir containing GDCMConfig.cmake (version subdir varies)
export G4DATA=/path/to/G4DATA                        # Geant4 runtime data
export SDK="$(xcrun --show-sdk-path)"
```

### (A) Patch the OpenTOPAS source

```bash
patch -p1 -d "$TOPAS_SRC" < "$MOP/topas_patches/opentopas_local.patch"
# If already applied it skips with "previously applied". To revert: add -R to the same command.
```

### (B) Metal shaders → metallib

```bash
mkdir -p "$MOP/build" && cd "$MOP/build"
for s in PhotonGeneration OpticalPhotonKernel DDAScoring; do
  xcrun metal -c "$MOP/shaders/$s.metal" -I "$MOP/include" -I "$MOP/shaders" -o "$s.air" -std=metal3.1
done
xcrun metallib PhotonGeneration.air OpticalPhotonKernel.air DDAScoring.air -o default.metallib
```

### (C) GPU engine dylib

```bash
cd "$MOP/build"
clang++ -std=c++17 -ObjC++ -fobjc-arc -O2 -c "$MOP/src/MetalOpticalEngine.mm" \
  -I "$MOP/include" -I "$MOP/topas_extension" -o MetalOpticalEngine.o
clang++ -std=c++17 -O2 -c "$MOP/topas_extension/TopasParameterParser.cc" \
  -I "$MOP/topas_extension" -I "$MOP/include" -o TopasParameterParser.o
clang++ -dynamiclib -O2 MetalOpticalEngine.o TopasParameterParser.o \
  -o libMetalOpticalPhoton.dylib \
  -framework Metal -framework MetalPerformanceShaders -framework Foundation \
  -install_name "@rpath/libMetalOpticalPhoton.dylib"
```

### (D) Build TOPAS + the GPU extension

```bash
mkdir -p "$TOPAS_BUILD" && cd "$TOPAS_BUILD"
export Geant4_DIR="$GEANT4_DIR"
cmake "$TOPAS_SRC" \
  -DCMAKE_INSTALL_PREFIX="$TOPAS_INSTALL" -DCMAKE_BUILD_TYPE=Release \
  -DTOPAS_EXTENSIONS_DIR="$MOP/topas_extension" \
  -DGeant4_DIR="$GEANT4_DIR/lib/cmake/Geant4" -DGDCM_DIR="$GDCM_DIR" \
  -DEXPAT_INCLUDE_DIR="$SDK/usr/include" -DZLIB_INCLUDE_DIR="$SDK/usr/include" \
  -DEXPAT_LIBRARY="$SDK/usr/lib/libexpat.tbd" -DZLIB_LIBRARY_RELEASE="$SDK/usr/lib/libz.tbd" \
  -DCMAKE_CXX_FLAGS="-include cmath -iquote $MOP/include -iquote $MOP/topas_extension" \
  -DCMAKE_EXE_LINKER_FLAGS="-L$MOP/build -lMetalOpticalPhoton -Wl,-rpath,$MOP/build -framework Metal -framework Foundation -framework MetalPerformanceShaders" \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DTOPAS_USE_QT=ON -DTOPAS_USE_QT6=ON
make -j"$(sysctl -n hw.ncpu)"     # use -j4 etc. to lower the job count
make install
```

> **Both flags in `-DCMAKE_CXX_FLAGS` are required:**
> - `-include cmath` : prevents the macOS SDK `<math.h>` `isinf` macro from clashing with libc++
>   `<complex>`'s `std::isinf`. Without it, compilation errors out.
> - `-iquote $MOP/include -iquote $MOP/topas_extension` : TOPAS copies the extension `.cc/.hh`
>   flattened into `build/extensions/`, so the wrapper headers' (`MOPTypes.hh`,
>   `MetalOpticalEngine.hh`) `#include "../include/MOPTypes.h"` would resolve to `build/include/`
>   (which doesn't exist) and die with **`'../include/MOPTypes.h' file not found`**. `-iquote`
>   puts the original `include/` · `topas_extension/` on the quoted-include search path so the
>   canonical headers resolve. Without it, the `extensions` target fails to build.
>
> If `make` fails, fix the cause and just continue (incrementally) with
> `cd "$TOPAS_BUILD" && make`.

### (E) Install the engine + generate the run wrapper

```bash
mkdir -p "$TOPAS_INSTALL/lib"
cp "$MOP/build/libMetalOpticalPhoton.dylib" "$TOPAS_INSTALL/lib/"
cp "$MOP/build/default.metallib"            "$TOPAS_INSTALL/lib/"

cat > "$TOPAS_INSTALL/bin/topas-gpu" <<EOF
#!/bin/bash
SCRIPT_DIR="\$(cd "\$(dirname "\$0")/.." && pwd)"
export QT_QPA_PLATFORM_PLUGIN_PATH="\$SCRIPT_DIR/Frameworks"
export TOPAS_G4_DATA_DIR="$G4DATA"
export DYLD_LIBRARY_PATH="\$SCRIPT_DIR/lib:\${DYLD_LIBRARY_PATH}"
export DYLD_LIBRARY_PATH="$GEANT4_DIR/lib:\${DYLD_LIBRARY_PATH}"
export METAL_DEVICE_WRAPPER_TYPE=1
export G4TRACE_DIR=OFF
exec "\$SCRIPT_DIR/bin/topas" "\$@"
EOF
chmod +x "$TOPAS_INSTALL/bin/topas-gpu"
```

Then run with `topas-gpu your_sim.txt`. (To rebuild only the engine/shaders, repeat just
(B)(C)(E) — no TOPAS rebuild needed.)
