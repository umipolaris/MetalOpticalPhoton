#ifndef COMMON_METAL_H
#define COMMON_METAL_H
/**
 * Common.h
 * 공통 타입, 유틸리티, 속성 보간 함수
 *
 * packed_float3 사용: CPU MOPTypes.h의 flat float 필드와 레이아웃 일치
 * (packed_float3 = 12바이트, padding 없음 → float x,y,z와 동일)
 */

#include <metal_stdlib>
using namespace metal;

#define MOP_SQRT(x)  metal::sqrt((float)(x))
#define MOP_SIN(x)   metal::sin((float)(x))
#define MOP_COS(x)   metal::cos((float)(x))
#define MOP_LOG(x)   metal::log((float)(x))
#define MOP_EXP(x)   metal::exp((float)(x))

// ============================================================
// GPU 측 상수 (MOPTypes.h와 동기화)
// ============================================================
constant uint MAX_PROPERTY_ENTRIES = 512;
constant uint MAX_MATERIALS = 64;
constant uint MAX_SURFACES = 128;
constant float PHOTON_ENERGY_MIN = 1.0;   // eV
constant float PHOTON_ENERGY_MAX = 15.0;  // eV

// Fix 64: 균일 LUT — InterpolateProperty 이진 탐색을 직접 인덱싱으로 대체
// LUT는 [PHOTON_ENERGY_MIN, PHOTON_ENERGY_MAX] 범위를 64등분한 균일 격자
constant uint LUT_SIZE = 256;  // BK7 굴절률 interp 정밀도
constant uint MAX_KNOTS = 64;   // absLen per-photon spline eval knot 최대 (MOP_MAX_KNOTS 와 동일)
constant uint LUT_PROPS_PER_MAT = 5;  // rindex, absLen, rayLen, mieLen, wlsLen
constant float LUT_ENERGY_STEP = (PHOTON_ENERGY_MAX - PHOTON_ENERGY_MIN) / (LUT_SIZE - 1);

// 재질당 LUT. absLen 슬롯은 G4 spline knot 패킹: [0]=N, [1..64]=E, [65..128]=V, [129..192]=D2.
struct MaterialLUT {
    float rindex[LUT_SIZE];
    float absLen[LUT_SIZE];   // packed spline knots
    float rayLen[LUT_SIZE];
    float mieLen[LUT_SIZE];
    float wlsLen[LUT_SIZE];
};

// 빠른 LUT 보간 — 2회 인접 읽기만 수행 (이진 탐색의 9회 산발 읽기 대체)
inline float InterpolateLUT(const device float* lut, float energy) {
    float t = (energy - PHOTON_ENERGY_MIN) / LUT_ENERGY_STEP;
    t = clamp(t, 0.0f, float(LUT_SIZE - 1));
    uint idx = min(uint(t), LUT_SIZE - 2);
    float frac = t - float(idx);
    return mix(lut[idx], lut[idx + 1], frac);
}

// 2026-05-24: absLen per-photon 정확 eval — G4 PhysicsVector Interpolation (icc:125) 와 동일.
// LUT 보간오차 0. packed absLen[]: [0]=N, [1..64]=E, [65..128]=V, [129..192]=D2.
// secD 는 host 에서 G4 ComputeSecDerivative1 (not-a-knot) 로 계산.
inline float SplineEvalG4Packed(const device float* p, float energy) {
    uint n = uint(p[0] + 0.5f);
    if (n == 0u) return 1e30f;        // abslen property 없음 → 흡수 없음
    if (n == 1u) return p[65];        // 상수 (V[0] at [65])
    const device float* E  = p + 1;
    const device float* V  = p + 65;
    const device float* D2 = p + 129;
    if (energy <= E[0]) return V[0];
    if (energy >= E[n-1u]) return V[n-1u];
    uint lo = 0u, hi = n - 1u;
    while (hi - lo > 1u) { uint mid = (lo+hi)/2u; if (E[mid] <= energy) lo = mid; else hi = mid; }
    float dl = E[hi] - E[lo];
    float b = (energy - E[lo]) / dl;
    float res = V[lo] + b * (V[hi] - V[lo]);
    float c0 = (2.0f - b) * D2[lo];
    float c1 = (1.0f + b) * D2[hi];
    res += (b * (b - 1.0f)) * (c0 + c1) * (dl * dl / 6.0f);
    return res;
}
constant float HC_EVNM = 1239.84198;     // hc in eV·nm
constant float SPEED_OF_LIGHT = 299.792458; // mm/ns
constant float PI = 3.14159265358979323846;
constant float TWO_PI = 6.28318530717958647692;
constant float FINE_STRUCTURE_ALPHA = 0.0072973525693; // α ≈ 1/137

// ============================================================
// 광자 상태
// ============================================================
enum PhotonStatus : uint {
    ALIVE        = 0,
    ABSORBED     = 1,
    DETECTED     = 2,
    BOUNDARY_ABS = 3,
    OUT_OF_WORLD = 4,
    MAX_STEPS    = 5,
    REEMITTED    = 6
};

enum SurfaceFinish : uint {
    POLISHED              = 0,
    POLISHED_FRONT_PAINT  = 1,
    POLISHED_BACK_PAINT   = 2,
    GROUND                = 3,
    GROUND_FRONT_PAINT    = 4,
    GROUND_BACK_PAINT     = 5
};

enum SurfaceType : uint {
    DIELECTRIC_DIELECTRIC = 0,
    DIELECTRIC_METAL      = 1
};

enum SurfaceModel : uint {
    GLISUR  = 0,
    UNIFIED = 1
};

enum GenType : uint {
    SCINTILLATION = 0,
    CERENKOV      = 1,
    WLS           = 2
};

// ============================================================
// GPU 데이터 구조체
// packed_float3 사용하여 CPU 측 flat float 필드와 바이트 레이아웃 일치
// ============================================================
struct PropertyTable {
    uint  count;
    float energies[MAX_PROPERTY_ENTRIES];
    float values[MAX_PROPERTY_ENTRIES];
};

struct MaterialGPU {
    uint materialId;
    uint isOpticalEnabled;

    PropertyTable refractiveIndex;
    PropertyTable absorptionLength;
    PropertyTable fastComponent;
    PropertyTable slowComponent;
    PropertyTable wlsAbsLength;
    PropertyTable wlsComponent;
    PropertyTable rayleighLength;
    PropertyTable mieScattering;

    float scintillationYield;
    float resolutionScale;
    float fastTimeConstant;
    float slowTimeConstant;
    float yieldRatio;
    float birksConstant;
    float wlsTimeConstant;
    float miehgForward;
    float miehgBackward;
    float miehgForwardRatio;
    uint  isDetector;
};

struct SurfaceGPU {
    uint surfaceId;
    uint type;
    uint finish;
    uint model;
    float sigmaAlpha;

    PropertyTable reflectivity;
    PropertyTable efficiency;
    PropertyTable transmittance;
    PropertyTable specularLobe;
    PropertyTable specularSpike;
    PropertyTable backScatter;
    // Force-reflectivity strict mode flag — TOPAS extension param
    // b:Su/XXX/ForceReflectivity = "True" 시 reflectivity LUT 값을
    // 절대 반사 확률로 사용 (Fresnel 우회).
    uint forceReflectivity;
    float _padding[1];
};

// Fix 71: 표면 LUT — SurfaceGPU의 PropertyTable 이진탐색을 직접 인덱싱으로 대체
// 표면당 6속성 × 64엔트리 = 384 floats = 1.5KB
struct SurfaceLUT {
    float reflectivity[LUT_SIZE];
    float efficiency[LUT_SIZE];
    float transmittance[LUT_SIZE];
    float specularLobe[LUT_SIZE];
    float specularSpike[LUT_SIZE];
    float backScatter[LUT_SIZE];
};

struct Genstep {
    packed_float3 position;
    float  time;
    packed_float3 direction;
    float  charge;
    uint   genType;
    uint   materialId;
    uint   numPhotons;
    uint   parentTrackId;
    float  betaInverse;
    float  pmin;
    float  pmax;
    float  maxCos;
    float  edep;
    float  stepLength;
    float  yieldRatio;
    float  _padding[1];
};

struct Photon {
    packed_float3 position;
    packed_float3 direction;
    packed_float3 polarization;
    float  wavelength;
    float  energy;
    float  time;
    float  weight;
    uint   status;
    uint   volumeId;
    uint   materialId;
    uint   stepCount;
    uint   flags;
    // 진단 (2026-04-22): GPU/CPU reflect 비교용 — 실제 reflect 분기 진입 횟수
    // (SameMaterial early-exit/직선 transmission 제외)
    uint   reflectedCount;
    // 2026-04-23 Chroma 정석 fix (L1 expert): 직전 BVH triangle ID 저장.
    // 다음 step BVH intersect 시 같은 triangle reject로 self-intersection 누설 차단.
    //
    // Lifecycle invariant (2026-04-27 U3 halo fix):
    //   Reset to -1 on:
    //     (a) photon termination (status != ALIVE) — implicit, photon discarded
    //     (b) ANY in-place scattering event that changes direction without crossing
    //         a boundary: Rayleigh, Mie, WLS reemission. After such events the photon
    //         legitimately may re-hit the previously-recorded triangle from a new
    //         direction. Failing to reset = stale dedup → BVH skips legitimate
    //         boundaries → distToBoundary jumps to far face → next-process winner
    //         can teleport photon ACROSS the boundary without boundary processing.
    //   Set to triangle.id on:
    //     (c) BVH accept (line 1038) — boundary crossing or triangle hit detected.
    //
    //   When adding a new in-place scattering process, MUST reset lastHitTriId after
    //   the process modifies photon.direction. See OpticalPhotonKernel.metal Rayleigh
    //   / Mie / WLS branches for examples (lines 1900, 1917, 1975).
    int    lastHitTriId;
    // Fix 78측정-6: genstepId/parentTrackId → PhotonMeta로 분리. 80B → 72B
};

// Fix 78측정-6: cold meta — propagate kernel은 접근 안함. hit/transit 기록 시만.
struct PhotonMeta {
    uint genstepId;
    uint parentTrackId;
};

struct Hit {
    packed_float3 position;
    float  time;
    float  energy;
    float  wavelength;
    uint   volumeId;
    uint   parentTrackId;
    uint   flags;
    uint   _padding[3];
};

// Fix 41: 스코어링 볼륨 통과 기록 (CPU Fluence 호환)
// 광자가 스코어링 볼륨에 진입할 때 위치+방향을 기록
// 광자를 죽이지 않고 계속 전파 (검출과 달리 단순 기록)
struct TransitHit {
    packed_float3 position;    // 진입 위치 (mm)
    packed_float3 direction;   // 진입 방향
    float  energy;             // 광자 에너지 (eV)
    float  weight;             // AABB 내 경로 길이 (mm) — geometric, weight 미포함
    float  photonWeight;       // 광자 가중치 (기본 1.0)
    uint   materialId;         // F2 정석 fix (2026-04-23): 광자 currentMatId.
                               // DDA scorer는 이 material에 누적되는 hit만 사용
                               // (Glass step이 Air scorer에 over-count되는 문제 방지)
};

constant uint MAX_VOLUMES = 64;   // Fix 41: 스코어링 볼륨 비트마스크 크기

// G1 (정석 fix, Opticks): G4Box analytic intersection
// boxGeometries 는 device buffer (개수 제한 없음). 활성 개수는 numBoxGeometries 로 전달.

// SF1 (2026-04-24): GPU surface flux scorer
// surfaceDefs 는 device buffer (개수 제한 없음). 활성 개수는 config.numSurfaces 로 전달.

// SF1: bounded plane surface 정의. AABB face는 axis-aligned bounded plane의 special case.
// 5명 합의: tagged union이지만 단일 layout으로 SIMT divergence 최소화.
struct SurfaceDef {
    packed_float3 origin;       // surface center point (mm)
    packed_float3 normal;       // unit normal (pointing "out" of host volume)
    packed_float3 axisU;        // in-plane axis 1 (unit)
    packed_float3 axisV;        // in-plane axis 2 (unit, ⟂ axisU)
    float halfExtentU;          // bounded extent along axisU (mm)
    float halfExtentV;          // bounded extent along axisV (mm)
    uint surfaceId;             // scorer 등록 순번 (0-based)
    uint enabled;               // 0/1 (registered by host scorer)
};

// SF1: surface crossing event 기록. 광자가 등록된 plane을 통과할 때마다 발생.
// 5명 합의 layout (64 bytes, 16B align):
//   pos[12] + dir[12] + energy[4] + cosθ[4] + weight[4] + time[4] = 40
//   surfaceId[4] + flags[4] + trackId[4] + matFrom[4] + matTo[4] + pad[4] = 24
struct SurfaceHit {
    packed_float3 position;     // crossing point (mm)
    packed_float3 direction;    // photon direction at crossing
    float energy;               // eV
    float cosTheta;             // dir · normal (signed, sign=in/out direction)
    float photonWeight;         // weight (default 1.0, for RR/splitting future)
    float time;                 // ns (TOF support)
    uint surfaceId;             // which registered surface
    uint flags;                 // bit 0: 1=going in (cosθ<0), 0=going out (cosθ≥0)
                                // bit 1: TIR before crossing, bit 2: s-pol component
    uint trackId;               // photon track ID (debug + multi-bounce filter)
    uint materialIdFrom;        // material on previous side
    uint materialIdTo;          // material on next side
    uint _pad;                  // 16B align
};

// G1: G4Box analytic intersection (MOPBoxGeometry 와 layout 동기화 필요)
struct BoxGeometry {
    float HLX;
    float HLY;
    float HLZ;
    packed_float3 center;
    uint  materialIdInside;
    uint  materialIdOutside;
    uint  surfaceId;
    uint  volumeId;
    uint  _pad[2];
};

// Phase 1 (2026-05-05): G4Sphere analytic intersection (MOPSphereGeometry layout)
// Phase 1.1 (2026-05-16): innerRadius (RMin>0) 지원 — sphere shell.
// Phase 1.2 (2026-05-16): theta range (STheta/DTheta) 지원 — hemisphere.
struct SphereGeometry {
    packed_float3 center;
    float         radius;
    float         innerRadius;  // RMin (0 = full sphere, >0 = shell)
    float         thetaStart;   // STheta (rad), 0 = north pole
    float         deltaTheta;   // DTheta (rad), π = full, π/2 = hemisphere
    uint          materialIdInside;
    uint          materialIdOutside;
    uint          surfaceId;
    uint          volumeId;
    uint          _pad[1];
};

// Phase 2 (2026-05-05): G4Tubs/TsCylinder analytic intersection (MOPCylinderGeometry layout)
// Z-axis aligned, full phi. Phase 2.1 (2026-05-16): innerRadius (tube/shell) 지원.
struct CylinderGeometry {
    packed_float3 center;
    float         radius;
    float         halfLength;
    float         innerRadius; // RMin (0 = full cyl, >0 = tube/shell)
    uint          materialIdInside;
    uint          materialIdOutside;
    uint          surfaceId;
    uint          volumeId;
    uint          endCapOpen;   // 1 = "구멍" cylinder: +z/-z plane open (Fix 33 analytic 등가)
    uint          _pad[1];
};

// Phase 3 (2026-05-05): G4Torus analytic intersection (SDF Sphere Tracing)
// Phase 3.5: generic axis (불일치 axis 시 world→local 회전). device buffer (개수 제한 없음).
struct TorusGeometry {
    packed_float3 center;
    float         rTor;
    float         rMax;
    packed_float3 axis;     // unit vector (world coord)
    uint          materialIdInside;
    uint          materialIdOutside;
    uint          surfaceId;
    uint          volumeId;
};

// Phase 4 (2026-05-18): G4Polyhedra (general N-sided) analytic intersection
// slab intersection — N lateral + 2 cap planes (partial phi 시 2 phi-cap 추가).
struct PolyhedraGeometry {
    // C struct mirror — 모든 field explicit float 로 alignment 명확화.
    float centerX, centerY, centerZ;     // world coord
    float axisX, axisY, axisZ;           // unit axis (world = local Z)
    float e1X, e1Y, e1Z;                 // local X in world
    float e2X, e2Y, e2Z;                 // local Y in world (RH: cross(axis,e1))
    float halfLengthAxis;
    float rMax;                          // circumscribed radius
    float rMin;                          // 0 = solid
    float phiStart;
    float phiTotal;
    uint  numSides;                      // N
    uint  materialIdInside;
    uint  materialIdOutside;
    uint  surfaceId;
    uint  volumeId;
    uint  _pad[3];
};

// Fix 61: 개별 광자 궤적 덤프 (GPU/CPU 경로 비교용)
// 다중 광자 dump 지원 — 처음 N개 광자의 모든 step을 photonIdx 와 함께 기록
constant uint MAX_TRAJECTORY_STEPS = 500000;
// 처음 이 개수 광자의 모든 step 을 dump (R_p≈0.85% 시나리오에서 first reflect 발견에 충분)
constant uint TRAJ_DUMP_PHOTON_LIMIT = 10000;

struct TrajectoryStep {
    packed_float3 position;     // 스텝 시작 위치
    packed_float3 direction;    // 스텝 시작 방향
    packed_float3 normal;       // 교차점 법선 (경계 히트 시)
    float stepDist;             // 실제 이동 거리
    float distToBoundary;       // HW RT 교차 거리
    float distToAbsorb;         // 흡수까지 거리
    float cosI;                 // 입사각 코사인
    float n1;                   // 입사 재질 굴절률
    float n2;                   // 투과 재질 굴절률
    float fresnelR;             // Fresnel 반사율
    float randVal;              // 반사/투과 결정에 사용된 난수
    uint  processType;          // 0=boundary, 1=absorb, 2=rayleigh, 3=mie, 4=wls, 5=worldExit
    uint  matIdIn;
    uint  matIdOut;
    uint  outcome;              // 0=transmit, 1=reflect, 2=TIR, 3=boundaryAbs, 4=bulkAbs
    int   triangleId;
    uint  volumeId;
    uint  photonIdx;            // 광자 ID (다중 광자 dump 시 그룹화용)
};

struct SimConfig {
    uint  maxStepsPerPhoton;
    uint  maxPhotonsPerBatch;
    uint  randomSeed;
    uint  enableScintillation;
    uint  enableCerenkov;
    uint  enableAbsorption;
    uint  enableRayleigh;
    uint  enableMie;
    uint  enableWLS;
    uint  enableBoundary;
    float worldSizeX;
    float worldSizeY;
    float worldSizeZ;
    uint  cerenkovMaxPhotonsPerStep;
    float cerenkovMaxBetaChange;
    uint  numGensteps;           // 실제 genstep 수 (생성 커널 루프 범위용)

    // Fix 41b: 스코어링 AABB (매 스텝마다 위치 체크)
    float scoringAABBMinX;
    float scoringAABBMinY;
    float scoringAABBMinZ;
    float scoringAABBMaxX;
    float scoringAABBMaxY;
    float scoringAABBMaxZ;
    uint  scoringAABBEnabled;    // AABB 기반 스코어링 활성화 여부

    // SF1 (2026-04-24): GPU surface flux scorer
    uint  numSurfaces;           // active surface scorer 수 (0 = 비활성)

    // 2026-05-08: BVH centroid offset (fp32 precision improvement)
    float bvhCentroidX;
    float bvhCentroidY;
    float bvhCentroidZ;
    // 2026-05-12 fix: per-batch RNG repetition. host 가 누적 photon 수 전달 →
    // InitRandomState(gid + globalPhotonOffset, ...) 로 batch 마다 unique RNG.
    uint  globalPhotonOffset;

    // 2026-05-17: GPU BEAM source 보조 (TOPAS Beam 정합).
    // 자세한 의미는 MOPTypes.h 의 MOPSimConfig 주석 참조.
    uint  beamPosShape;        // 0=Rectangle, 1=Ellipse
    float beamPosSigmaX;       // Gaussian sigma (mm); 0=Flat uniform
    float beamPosSigmaY;
    uint  beamPolMode;         // 0=random transverse, 1=fixed
    float beamPolX;
    float beamPolY;
    float beamPolZ;
    // 2026-05-17 추가 — 자세한 의미는 MOPTypes.h 참조
    uint  beamPosDist;          // 0=None, 1=Flat, 2=Gaussian
    uint  beamAngDist;          // 0=None, 1=Flat, 2=Gaussian, 3=Isotropic(4π)
    float beamEnergySpreadEv;   // Gaussian σ (eV); 0=mono
    float beamTimeSpread;       // Gaussian σ (ns); 0=t=0
    float beamTimeCutoff;       // abs cutoff (ns); 0=no clip
    // 2026-05-17 BeamEnergySpectrum (inline LUT). 자세한 의미는 MOPTypes.h 참조.
    uint  beamSpectrumNumBins;
    uint  beamSpectrumType;     // 0=Discrete, 1=Continuous
    float beamSpectrumEnergies[256];
    float beamSpectrumCumWeights[256];
    // 2026-05-19: raw weights for piecewise linear inverse CDF (Continuous only)
    float beamSpectrumWeights[256];
};

// ============================================================
// 삼각형 메시
// ============================================================
struct Triangle {
    packed_float3 v0, v1, v2;
    packed_float3 normal;
    uint   volumeId;
    uint   materialIdInside;
    uint   materialIdOutside;
    uint   surfaceId;
};

// Fix 71: 경량 삼각형 속성 — 커널에서 필요한 필드만 (32B, Triangle 64B의 절반)
// v0,v1,v2는 HW RT용 vertexPositionBuffer에만 존재, 커널에서는 미사용 (진단 제외)
struct TriangleAttr {
    // 2026-05-15: H7 vertex (v0/v1/v2) 제거 — shader read 0회 dead code, 68B→32B.
    // boundary step 마다 device-load 하는 struct 라 cacheline traffic 절반.
    packed_float3 normal;       // 12B
    uint   volumeId;            // 4B
    uint   materialIdInside;    // 4B
    uint   materialIdOutside;   // 4B
    uint   surfaceId;           // 4B
    uint   _pad;                // 4B → 총 32B
};

// ============================================================
// 속성 테이블 보간 (에너지 기반 선형 보간)
// device 주소 공간 버전
// ============================================================
inline float InterpolateProperty(const device PropertyTable& table, float energy) {
    if (table.count == 0) return 0.0;
    if (table.count == 1) return table.values[0];

    // 범위 밖 → 클램프
    if (energy <= table.energies[0]) return table.values[0];
    if (energy >= table.energies[table.count - 1]) return table.values[table.count - 1];

    // 이진 탐색으로 구간 찾기
    uint lo = 0, hi = table.count - 1;
    while (hi - lo > 1) {
        uint mid = (lo + hi) / 2;
        if (table.energies[mid] <= energy)
            lo = mid;
        else
            hi = mid;
    }

    // 선형 보간
    float e0 = table.energies[lo];
    float e1 = table.energies[hi];
    float v0 = table.values[lo];
    float v1 = table.values[hi];

    // Fix: 중복 에너지 엔트리로 인한 NaN 방지 (0으로 나누기 보호)
    float dE = e1 - e0;
    if (dE < 1e-20) return 0.5 * (v0 + v1);

    float t = (energy - e0) / dE;
    return mix(v0, v1, t);
}

// ============================================================
// CLHEP RanecuEngine 포팅 (L'Ecuyer 1988 CACM CMRG)
// TOPAS/Geant4가 사용하는 RanecuEngine과 동일 알고리즘.
// State 2 int32, 주기 ≈ 2.3e18, BigCrush all-pass.
// 검증 14: Philox와 비교용 재포팅.
// ============================================================
struct RandomState {
    int seed1;
    int seed2;
};

inline float RandUniform(thread RandomState& state) {
    constexpr int ecuyer_a = 40014;
    constexpr int ecuyer_b = 53668;
    constexpr int ecuyer_c = 12211;
    constexpr int ecuyer_d = 40692;
    constexpr int ecuyer_e = 52774;
    constexpr int ecuyer_f = 3791;
    constexpr int shift1   = 2147483563;
    constexpr int shift2   = 2147483399;
    constexpr float prec   = 4.6566128e-10f;

    int k1 = state.seed1 / ecuyer_b;
    int k2 = state.seed2 / ecuyer_e;

    state.seed1 = ecuyer_a * (state.seed1 - k1 * ecuyer_b) - k1 * ecuyer_c;
    if (state.seed1 < 0) state.seed1 += shift1;

    state.seed2 = ecuyer_d * (state.seed2 - k2 * ecuyer_e) - k2 * ecuyer_f;
    if (state.seed2 < 0) state.seed2 += shift2;

    int diff = state.seed1 - state.seed2;
    if (diff <= 0) diff += (shift1 - 1);

    return (float)diff * prec;
}

inline uint splitmix32(uint x) {
    x = (x ^ (x >> 16)) * 0x85ebca6bu;
    x = (x ^ (x >> 13)) * 0xc2b2ae35u;
    x = x ^ (x >> 16);
    return x;
}

inline RandomState InitRandomState(uint photonId, uint globalSeed, uint kernelSalt) {
    RandomState s;
    uint h1 = splitmix32(photonId * 0x9E3779B9u ^ globalSeed ^ kernelSalt);
    uint h2 = splitmix32(photonId * 0x85EBCA6Bu ^ globalSeed ^ (kernelSalt + 1u));
    s.seed1 = (int)(h1 % 2147483562u) + 1;
    s.seed2 = (int)(h2 % 2147483398u) + 1;
    return s;
}

inline float RandUniformRange(thread RandomState& state, float lo, float hi) {
    return lo + (hi - lo) * RandUniform(state);
}

// 반-가우시안 샘플링 (Box-Muller, α ≥ 0 only)
// Geant4 UNIFIED 모델 호환: sigmaAlpha에 대한 가우시안 분포
inline float RandHalfGaussian(thread RandomState& state, float sigma) {
    if (sigma <= 0.0) return 0.0;
    // Box-Muller 변환으로 가우시안 |N(0, sigma)| 샘플링
    float u1 = max(RandUniform(state), 1e-10f);  // log(0) 방지
    float u2 = RandUniform(state);
    float g = sigma * MOP_SQRT(-2.0 * MOP_LOG(u1)) * MOP_COS(TWO_PI * u2);
    return abs(g);  // 반-가우시안: 양수만
}

// 전-가우시안 N(0, sigma) — G4 UNIFIED 의 G4RandGauss::shoot(0, σα) 동등.
// 음수 / 양수 모두 가능. G4 의 GetFacetNormal sin(α) rejection 이 음수 α
// 거부하나 sample 자체는 full distribution 이어야 함.
inline float RandGauss(thread RandomState& state, float sigma) {
    if (sigma <= 0.0) return 0.0;
    float u1 = max(RandUniform(state), 1e-10f);
    float u2 = RandUniform(state);
    return sigma * MOP_SQRT(-2.0 * MOP_LOG(u1)) * MOP_COS(TWO_PI * u2);
}

// 코사인 가중 랜덤 방향 (Lambert) — Malley's method
// 2026-05-14: fp32 catastrophic cancellation 회피 ((1-u1) 직접 sample).
// 이전 sqrt(1.0 - u1) 은 u1 이 1 가까울 때 (1.0f - u1) 가 0 round → grazing
// reflect 광자의 normal component 손실. precise:: variant 는 효과 없어 미적용.
inline float3 RandLambertian(thread RandomState& state, float3 normal) {
    float oneMinusU1 = RandUniform(state);   // = (1 - u1) 직접 sample (uniform 동일 분포)
    float u2 = RandUniform(state);
    float u1 = 1.0f - oneMinusU1;

    float r = MOP_SQRT(u1);                  // disk radius = sin θ
    float theta = TWO_PI * u2;

    // 법선 기준 로컬 좌표계 생성
    float3 up = abs(normal.y) < 0.999 ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 tangent = normalize(cross(up, normal));
    float3 bitangent = cross(normal, tangent);

    return normalize(tangent * r * MOP_COS(theta) + bitangent * r * MOP_SIN(theta) +
                     normal * MOP_SQRT(oneMinusU1));   // normal comp = cos θ (cancellation-free)
}

#endif // COMMON_METAL_H
