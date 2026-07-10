/**
 * MOPTypes.h
 * MetalOpticalPhoton - GPU-accelerated optical photon propagation for TOPAS/Geant4
 *
 * 공통 데이터 타입 정의
 * TOPAS 파라미터 파일의 광학 속성을 GPU로 전달하기 위한 구조체들
 */

#ifndef MOP_TYPES_H
#define MOP_TYPES_H

#include <stdint.h>

#ifdef __METAL_VERSION__
    // Metal shader 내부
    #include <metal_stdlib>
    using namespace metal;
    #define MOP_ALIGNED(x) // Metal은 자체 정렬 규칙 사용
#else
    // CPU 측 (C/C++/Obj-C++)
    #ifdef __cplusplus
    extern "C" {
    #endif
    #define MOP_ALIGNED(x) // GPU 레이아웃 일치를 위해 정렬 패딩 제거
#endif

// ============================================================
// 최대값 상수
// ============================================================
#define MOP_MAX_MATERIALS          64
#define MOP_MAX_SURFACES           128
#define MOP_MAX_PROPERTY_ENTRIES   512   // 에너지 의존 속성의 최대 데이터 포인트 (Exp B: 300pt 다운샘플링 제거)
#define MOP_MAX_VOLUMES            1024
#define MOP_MAX_TRIANGLES_PER_VOL  65536
#define MOP_MAX_PHOTONS_PER_BATCH  (1 << 27)  // ~128M photons per GPU batch (Fix 68: 대형 배치)

// ============================================================
// 광자 상태 플래그
// ============================================================
typedef enum {
    MOP_PHOTON_ALIVE        = 0,
    MOP_PHOTON_ABSORBED     = 1,
    MOP_PHOTON_DETECTED     = 2,
    MOP_PHOTON_BOUNDARY_ABS = 3,
    MOP_PHOTON_OUT_OF_WORLD = 4,
    MOP_PHOTON_MAX_STEPS    = 5,
    MOP_PHOTON_REEMITTED    = 6
} MOPPhotonStatus;

// ============================================================
// 표면 마감 타입 (TOPAS Su/XXX/Finish)
// ============================================================
typedef enum {
    MOP_FINISH_POLISHED              = 0,
    MOP_FINISH_POLISHED_FRONT_PAINT  = 1,
    MOP_FINISH_POLISHED_BACK_PAINT   = 2,
    MOP_FINISH_GROUND                = 3,
    MOP_FINISH_GROUND_FRONT_PAINT    = 4,
    MOP_FINISH_GROUND_BACK_PAINT     = 5
} MOPSurfaceFinish;

// ============================================================
// 표면 타입 (TOPAS Su/XXX/Type)
// ============================================================
typedef enum {
    MOP_SURFACE_DIELECTRIC_DIELECTRIC = 0,
    MOP_SURFACE_DIELECTRIC_METAL      = 1
} MOPSurfaceType;

// ============================================================
// 표면 모델 (TOPAS Su/XXX/Model)
// ============================================================
typedef enum {
    MOP_MODEL_GLISUR  = 0,
    MOP_MODEL_UNIFIED = 1
} MOPSurfaceModel;

// ============================================================
// 광자 생성 타입
// ============================================================
typedef enum {
    MOP_GEN_SCINTILLATION = 0,
    MOP_GEN_CERENKOV      = 1,
    MOP_GEN_WLS           = 2,  // Wavelength Shifting
    // 2026-05-12: kernel-side beam photon generation. 1 genstep → numPhotons photons.
    // TOPAS Beam source 우회 → G4 event loop bottleneck 제거.
    MOP_GEN_BEAM          = 3
} MOPGenerationType;

// ============================================================
// 에너지 의존 속성 테이블 (GPU 텍스처에 매핑)
// TOPAS의 dv:/uv: 벡터 파라미터에 대응
// ============================================================
typedef struct MOP_ALIGNED(16) {
    uint32_t count;                          // 데이터 포인트 수
    float    energies[MOP_MAX_PROPERTY_ENTRIES]; // eV 단위 에너지 배열
    float    values[MOP_MAX_PROPERTY_ENTRIES];   // 속성값 배열
} MOPPropertyTable;

// ============================================================
// 물질 광학 속성 (TOPAS Ma/XXX/ 파라미터에 대응)
// ============================================================
typedef struct MOP_ALIGNED(16) {
    uint32_t materialId;
    uint32_t isOpticalEnabled;   // b:Ma/XXX/EnableOpticalProperties

    // === 벡터 속성 (에너지 의존) ===
    MOPPropertyTable refractiveIndex;    // dv:Ma/XXX/RefractiveIndex/Energies + uv:Values
    MOPPropertyTable absorptionLength;   // dv:Ma/XXX/AbsLength/Energies + uv:Values
    MOPPropertyTable fastComponent;      // uv:Ma/XXX/FastComponent (발광 스펙트럼)
    MOPPropertyTable slowComponent;      // uv:Ma/XXX/SlowComponent
    MOPPropertyTable wlsAbsLength;       // uv:Ma/XXX/WLSAbsLength (파장변환 흡수)
    MOPPropertyTable wlsComponent;       // uv:Ma/XXX/WLSComponent (파장변환 방출)
    MOPPropertyTable rayleighLength;     // uv:Ma/XXX/RayleighLength (레일리 산란 길이)
    MOPPropertyTable mieScattering;      // uv:Ma/XXX/Miehg

    // === 스칼라 속성 ===
    float scintillationYield;  // u:Ma/XXX/ScintillationYield (ph/MeV)
    float resolutionScale;     // u:Ma/XXX/ResolutionScale
    float fastTimeConstant;    // d:Ma/XXX/FastTimeConstant (ns)
    float slowTimeConstant;    // d:Ma/XXX/SlowTimeConstant (ns)
    float yieldRatio;          // u:Ma/XXX/YieldRatio (fast/total)
    float birksConstant;       // u:Ma/XXX/BirksConstant (mm/MeV)
    float wlsTimeConstant;     // d:Ma/XXX/WLSTimeConstant (ns)

    // Mie 산란 파라미터
    float miehgForward;        // u:Ma/XXX/MiehgForward
    float miehgBackward;       // u:Ma/XXX/MiehgBackward
    float miehgForwardRatio;   // u:Ma/XXX/MiehgForwardRatio

    uint32_t isDetector;       // 이 물질이 검출기인지 여부 (흡수 시 DETECTED 처리)
} MOPMaterialProperties;

// ============================================================
// 표면 광학 속성 (TOPAS Su/XXX/ 파라미터에 대응)
// ============================================================
typedef struct MOP_ALIGNED(16) {
    uint32_t surfaceId;
    uint32_t type;              // MOPSurfaceType  - s:Su/XXX/Type
    uint32_t finish;            // MOPSurfaceFinish - s:Su/XXX/Finish
    uint32_t model;             // MOPSurfaceModel  - s:Su/XXX/Model

    float    sigmaAlpha;        // u:Su/XXX/SigmaAlpha

    // 에너지 의존 표면 속성
    MOPPropertyTable reflectivity;         // uv:Su/XXX/Reflectivity
    MOPPropertyTable efficiency;           // uv:Su/XXX/Efficiency
    MOPPropertyTable transmittance;        // uv:Su/XXX/Transmittance
    MOPPropertyTable specularLobe;         // uv:Su/XXX/SpecularLobeConstant
    MOPPropertyTable specularSpike;        // uv:Su/XXX/SpecularSpikeConstant
    MOPPropertyTable backScatter;          // uv:Su/XXX/BackScatterConstant

    // Force-reflectivity strict mode (TOPAS extension):
    //   b:Su/XXX/ForceReflectivity = "True"
    // 0 = Geant4 default — surface.reflectivity 는 Fresnel-gate 확률 (rand<R 인
    //     경우 Fresnel 처리, 아니면 absorption/transmission)
    // 1 = Strict override — surface.reflectivity 는 절대 반사 확률 (rand<R →
    //     specular reflection, 아니면 직선 transmission, Fresnel 우회)
    uint32_t forceReflectivity;

    float _padding[1];
} MOPSurfaceProperties;

// ============================================================
// Genstep: Geant4에서 수집한 광자 생성 정보
// (Opticks 패턴을 Metal용으로 재구현)
// ============================================================
typedef struct MOP_ALIGNED(16) {
    // 생성 위치 및 방향
    float posX, posY, posZ;     // 생성 위치 (mm)
    float time;                 // 생성 시간 (ns)

    float dirX, dirY, dirZ;     // 입자 방향 (체렌코프용)
    float charge;               // 입자 전하

    // 생성 파라미터
    uint32_t genType;           // MOPGenerationType
    uint32_t materialId;        // 생성 위치의 물질 ID
    uint32_t numPhotons;        // 이 스텝에서 생성할 광자 수
    uint32_t parentTrackId;     // 부모 입자 트랙 ID

    // 체렌코프 전용
    float betaInverse;          // 1/β
    float pmin, pmax;           // 광자 에너지 범위 (eV)
    float maxCos;               // 최대 cos(θ_c)

    // 신틸레이션 전용
    float edep;                 // 에너지 침적 (MeV)
    float stepLength;           // 스텝 길이 (mm)
    float yieldRatio;           // fast/total 비율

    float _padding[1];
} MOPGenstep;

// ============================================================
// 광자 상태 (GPU에서 전파 중인 광자)
// ============================================================
typedef struct MOP_ALIGNED(16) {
    // 위치 및 방향
    float posX, posY, posZ;
    float dirX, dirY, dirZ;

    // 편광
    float polX, polY, polZ;

    // 물리량
    float wavelength;           // nm
    float energy;               // eV
    float time;                 // ns
    float weight;               // 통계적 가중치

    // 상태 추적
    uint32_t status;            // MOPPhotonStatus
    uint32_t volumeId;          // 현재 볼륨 ID
    uint32_t materialId;        // 현재 물질 ID
    uint32_t stepCount;         // 현재까지의 스텝 수
    uint32_t flags;             // 비트 플래그 (경계 반사 횟수 등)
    // 진단 (2026-04-22): GPU/CPU reflect 비교용 — 실제 reflect 분기 진입 횟수
    // (SameMaterial early-exit, transmission 직선 통과는 제외)
    uint32_t reflectedCount;    // ProcessBoundary{Fresnel,WithSurface}에서 실제 reflect/TIR 분기 진입 카운트
    // 2026-04-23 Chroma 정석 fix (L1): self-intersection 누설 차단용 직전 hit triangle ID.
    // Lifecycle: reset to -1 after any in-place scattering (Rayleigh/Mie/WLS) — photon's
    // geometric relationship to triangles resets when direction changes via scattering.
    // Failing to reset = stale dedup permanently blocks legitimate boundaries → next-process
    // winner can teleport photon across the boundary (U3 halo +78% bias 2026-04-27 fix).
    // See shaders/Common.h Photon struct for full lifecycle invariant.
    int32_t  lastHitTriId;
    // Fix 78측정-6: genstepId/parentTrackId를 별도 PhotonMeta 버퍼로 분리 (SOA hot/cold split)
} MOPPhoton;

// Fix 78측정-6: cold meta — hit 기록 시점에만 접근. propagate kernel hot path 외부.
typedef struct MOP_ALIGNED(8) {
    uint32_t genstepId;         // 원래 genstep ID
    uint32_t parentTrackId;     // 부모 트랙 ID
} MOPPhotonMeta;

// ============================================================
// 광자 히트 (검출기에 도달한 광자)
// ============================================================
typedef struct MOP_ALIGNED(16) {
    float posX, posY, posZ;
    float time;
    float energy;               // eV
    float wavelength;           // nm
    uint32_t volumeId;          // 검출 볼륨 ID
    uint32_t parentTrackId;
    uint32_t flags;
    uint32_t _padding[3];
} MOPHit;

// ============================================================
// Fix 61: 궤적 덤프 스텝 (GPU/CPU 경로 비교용)
// ============================================================
typedef struct {
    float posX, posY, posZ;
    float dirX, dirY, dirZ;
    float normX, normY, normZ;
    float stepDist;
    float distToBoundary;
    float distToAbsorb;
    float cosI;
    float n1, n2;
    float fresnelR;
    float randVal;
    uint32_t processType;     // 0=boundary, 1=absorb, 2=rayleigh, 3=mie
    uint32_t matIdIn, matIdOut;
    uint32_t outcome;          // 0=transmit, 1=reflect, 2=TIR, 3=bndAbs, 4=bulkAbs
    int32_t triangleId;
    uint32_t volumeId;
    uint32_t photonIdx;        // 광자 ID (다중 광자 dump 시 그룹화용)
} MOPTrajectoryStep;

// ============================================================
// Fix 65: GPU DDA 스코어링 설정
// 2026-05-16 (Phase 2.1+): cylindrical/spherical voxelization 지원
//   voxelType=0 (BOX): compFullX/Y/Z = Cartesian widths, bins X/Y/Z = Cartesian
//   voxelType=1 (CYLINDER): dim 0 = R (compFullX = RMax-RMin), 1 = Phi (rad),
//                            2 = Z (mm). rMin/rMax = inner/outer radial range,
//                            zHL = Z half-length. phiStart = lower bound (rad).
//   voxelType=2 (SPHERE):  dim 0 = R, 1 = Theta (rad), 2 = Phi (rad).
//                            rMin/rMax = radial range. thetaStart/phiStart =
//                            lower bounds.
// ============================================================
typedef struct MOP_ALIGNED(16) {
    float compTransX, compTransY, compTransZ;  //  0-11  컴포넌트 월드 위치 (mm)
    float compFullX, compFullY, compFullZ;     // 12-23  voxel grid 폭 (mm/rad)
    uint32_t nBinsX, nBinsY, nBinsZ;           // 24-35
    uint32_t totalTransitHits;                  // 36-39
    float binVolume;                            // 40-43  (BOX: dx*dy*dz, CYL/SPH: bin-마다 다름, host 평균값)
    float fluenceScale;                         // 44-47
    uint32_t scoringMaterialId;                 // 48-51
    uint32_t voxelType;                         // 52-55  0=BOX 1=CYLINDER 2=SPHERE
    float rMin;                                 // 56-59  CYL/SPH inner radius (mm)
    float phiStart;                             // 60-63  CYL Phi / SPH Phi 시작 각 (rad), default 0
    float thetaStart;                           // 64-67  SPH Theta 시작 각 (rad), default 0
    float energyLow;                            // 68-71  energy filter lower bound (eV), 0 = no lower bound
    float energyHigh;                           // 72-75  energy filter upper bound (eV), 0 or large = no upper bound
    // 2026-05-20: wavelength/energy binning (CPU Fluence EBins 동등). nBinsE=1 = 기존 동작.
    // binBuffer layout: [eBin * totalVoxels + voxel]. eBin = (energy-eMinEv)/(eMaxEv-eMinEv)*nBinsE
    uint32_t nBinsE;                            // 76-79  energy bins (1 = no E binning)
    float eMinEv;                               // 80-83  E bin lower (eV)
    float eMaxEv;                               // 84-87  E bin upper (eV)
    uint32_t _pad[2];                           // 88-95  16B align
} MOPDDAConfig;

// ============================================================
// Fix 64: 재질 LUT (InterpolateProperty 이진 탐색 대체)
// ============================================================
#define MOP_LUT_SIZE 256
// 2026-05-24: absLen 슬롯(256 float)에 G4 spline knot 을 패킹 (struct 변경 회피).
//   layout: [0]=N, [1..64]=E(eV), [65..128]=V(mm), [129..192]=D2. (1+3*64=193 <= 256)
#define MOP_MAX_KNOTS 64

typedef struct {
    float rindex[MOP_LUT_SIZE];
    float absLen[MOP_LUT_SIZE];   // packed spline knots (위 layout)
    float rayLen[MOP_LUT_SIZE];
    float mieLen[MOP_LUT_SIZE];
    float wlsLen[MOP_LUT_SIZE];
} MOPMaterialLUT;

// Fix 71: 표면 LUT — SurfaceGPU PropertyTable binary search 제거
typedef struct {
    float reflectivity[MOP_LUT_SIZE];
    float efficiency[MOP_LUT_SIZE];
    float transmittance[MOP_LUT_SIZE];
    float specularLobe[MOP_LUT_SIZE];
    float specularSpike[MOP_LUT_SIZE];
    float backScatter[MOP_LUT_SIZE];
} MOPSurfaceLUT;

// ============================================================
// 삼각형 메시 (Metal Ray Tracing 가속 구조에 사용)
// ============================================================
typedef struct MOP_ALIGNED(16) {
    float v0x, v0y, v0z;       // 꼭짓점 0
    float v1x, v1y, v1z;       // 꼭짓점 1
    float v2x, v2y, v2z;       // 꼭짓점 2
    float nx, ny, nz;          // 면 법선
    uint32_t volumeId;         // 소속 볼륨
    uint32_t materialIdInside; // 안쪽 물질
    uint32_t materialIdOutside;// 바깥쪽 물질
    uint32_t surfaceId;        // 표면 속성 ID (0 = 없음)
} MOPTriangle;

// ============================================================
// G1 (정석 fix, Opticks reference): G4Box analytic slab intersection.
// CPU G4Box::DistanceToOut/In 과 동일. BVH float32 mesh 정밀도 손실 우회.
// axis-aligned + 회전 없는 G4Box 만 지원 (회전된 box 는 BVH-only fallback).
// ============================================================
typedef struct MOP_ALIGNED(16) {
    float    HLX, HLY, HLZ;
    float    centerX, centerY, centerZ;
    uint32_t materialIdInside;
    uint32_t materialIdOutside;
    uint32_t surfaceId;
    uint32_t volumeId;
    uint32_t _pad[2];
} MOPBoxGeometry;

// ============================================================
// Phase 1 (2026-05-05, Opticks 정석): G4Sphere analytic intersection.
// CPU G4Sphere::DistanceToOut/In (quadratic at²+bt+c=0) 와 동일한 수식을
// Metal RT 의 Custom Intersection Function 으로 GPU 에서 직접 계산.
// Phase 1.1 (2026-05-16): innerRadius (RMin>0) 지원 — sphere shell.
// Phase 1.2 (2026-05-16): theta range (STheta/DTheta) 지원 — hemisphere.
//   광자 hit 점의 theta 가 [thetaStart, thetaStart+deltaTheta] 안인지 검사.
//   theta-cone cap surface intersection 은 미처리 (같은 material trivial
//   boundary case 만 작동). full DPhi=360 만 (partial phi 는 mesh fallback).
// ============================================================
typedef struct MOP_ALIGNED(16) {
    float    centerX, centerY, centerZ;  // sphere center (world coord, mm)
    float    radius;                     // RMax (mm)
    float    innerRadius;                // RMin (mm). 0 = full sphere, >0 = shell
    float    thetaStart;                 // STheta (rad). 0 = north pole
    float    deltaTheta;                 // DTheta (rad). π = full sphere, π/2 = hemisphere
    uint32_t materialIdInside;
    uint32_t materialIdOutside;
    uint32_t surfaceId;                  // border/skin surface (0 = none)
    uint32_t volumeId;
    uint32_t _pad[1];                    // 16B align (40 bytes round up to 48)
} MOPSphereGeometry;

// ============================================================
// Phase 2 (2026-05-05, Opticks 정석): G4Tubs/TsCylinder analytic intersection.
// Z-axis aligned cylinder (default G4Tubs orientation), full phi.
// Phase 2.1 (2026-05-16): innerRadius (RMin>0) 지원 → tube/shell 형태.
// 회전, partial phi 는 여전히 mesh fallback.
// ============================================================
typedef struct MOP_ALIGNED(16) {
    float    centerX, centerY, centerZ;  // cylinder center (world coord, mm)
    float    radius;                     // RMax (mm)
    float    halfLength;                 // HL — z 방향 half-length (mm)
    float    innerRadius;                // RMin (mm). 0 = full cylinder, >0 = tube shell
    uint32_t materialIdInside;
    uint32_t materialIdOutside;
    uint32_t surfaceId;
    uint32_t volumeId;
    uint32_t endCapOpen;                 // 1 = "구멍" cylinder (matId==grandparentMatId): +z/-z plane open
    uint32_t _pad[1];                    // 16B align
} MOPCylinderGeometry;

// ============================================================
// Phase 3 (2026-05-05): G4Torus analytic intersection (SDF Sphere Tracing).
// Phase 3.5 (2026-05-05): Generic-axis 지원 — torus axis 가 임의 방향이어도
// world→local 회전 후 SDF 계산 + local→world normal 변환.
// Full phi, RMin=0 만 지원. 다른 parameter → mesh fallback.
// ============================================================
typedef struct MOP_ALIGNED(16) {
    float    centerX, centerY, centerZ;
    float    rTor;     // major radius (torus axis to tube center)
    float    rMax;     // minor radius (tube cross-section)
    float    axisX, axisY, axisZ;   // unit axis vector (world coord)
    uint32_t materialIdInside;
    uint32_t materialIdOutside;
    uint32_t surfaceId;
    uint32_t volumeId;
} MOPTorusGeometry;

// ============================================================
// Phase 4 (2026-05-18, next session): G4Polyhedra (general N-sided) analytic.
// CSG = convex N-sided regular polygon cross-section × axis-direction extent.
// arbitrary axis direction in world coord.
//
// Intersection: slab approach
//   - N lateral half-spaces (one per side, normal in local XY plane)
//   - 2 axial cap half-spaces (Z = ±halfLengthAxis in local frame)
//   - partial phi (phiTotal < 2π): 2 phi-cap half-spaces (start/end angle)
//   - tEnter = max(entry t's), tExit = min(exit t's), valid if tEnter < tExit
//
// Each lateral side i:
//   angle_i = phiStart + (i + 0.5) × (phiTotal/N)
//   normal (local XY) = (cos angle_i, sin angle_i, 0)
//   plane offset (apothem) = RMax × cos(π/N)
//
// MVP: simple RZ profile (rectangular = single radius prism) + full phi.
//      RMin=0 (no shell). 추후 일반화.
// ============================================================
typedef struct MOP_ALIGNED(16) {
    float    centerX, centerY, centerZ;     // prism geometric center (world coord, mm)
    float    axisX, axisY, axisZ;            // prism axis unit vector (world coord) — local Z
    float    e1X, e1Y, e1Z;                  // local X axis in world coord
    float    e2X, e2Y, e2Z;                  // local Y axis in world coord (cross(axis,e1) — RH)
    float    halfLengthAxis;                 // axis-direction half-length (mm)
    float    rMax;                           // circumscribed radius (mm)
    float    rMin;                           // inner radius (0 = solid, >0 = shell, 미지원)
    float    phiStart;                       // local frame phi range start (rad)
    float    phiTotal;                       // local frame phi total (rad, 2π = full)
    uint32_t numSides;                       // N (3=triangle, 4=square, 6=hex, ...)
    uint32_t materialIdInside;
    uint32_t materialIdOutside;
    uint32_t surfaceId;
    uint32_t volumeId;
    uint32_t _pad[3];                        // 16B align
} MOPPolyhedraGeometry;

// ============================================================
// GPU 시뮬레이션 설정
// ============================================================
typedef struct MOP_ALIGNED(16) {
    uint32_t maxStepsPerPhoton;      // 광자당 최대 전파 스텝
    uint32_t maxPhotonsPerBatch;     // 배치당 최대 광자 수
    uint32_t randomSeed;             // 난수 시드
    uint32_t enableScintillation;    // 신틸레이션 활성화
    uint32_t enableCerenkov;         // 체렌코프 활성화
    uint32_t enableAbsorption;       // 흡수 활성화
    uint32_t enableRayleigh;         // 레일리 산란 활성화
    uint32_t enableMie;              // 미 산란 활성화
    uint32_t enableWLS;              // 파장변환 활성화
    uint32_t enableBoundary;         // 경계 처리 활성화
    float    worldSizeX;             // 월드 크기 (mm)
    float    worldSizeY;
    float    worldSizeZ;
    uint32_t cerenkovMaxPhotonsPerStep; // 체렌코프 스텝당 최대 광자 수
    float    cerenkovMaxBetaChange;     // 체렌코프 최대 β 변화율
    uint32_t numGensteps;               // GPU에 전달된 genstep 수 (생성 커널 루프 범위용)

    // Fix 41b: 스코어링 AABB (매 스텝마다 위치 체크)
    float scoringAABBMinX;
    float scoringAABBMinY;
    float scoringAABBMinZ;
    float scoringAABBMaxX;
    float scoringAABBMaxY;
    float scoringAABBMaxZ;
    uint32_t scoringAABBEnabled;        // AABB 기반 스코어링 활성화 여부

    // SF1 (2026-04-24): GPU surface flux scorer
    uint32_t numSurfaces;               // active surface scorer 수 (Common.h SimConfig와 동기화)

    // 2026-05-08: BVH centroid offset for fp32 precision improvement at oblique
    // mesh refraction (e.g., torus+lens at z=-300mm). 모든 mesh vertex 를 centroid
    // 만큼 -로 shift 후 BVH build → 광자 ray cast 시 photon.position - centroid 적용.
    // fp32 ULP 가 |coord| 비례하므로 centroid 가까울수록 정밀도 향상.
    float bvhCentroidX;
    float bvhCentroidY;
    float bvhCentroidZ;
    // 2026-05-12 fix: per-batch RNG repetition. host 누적 photon offset.
    uint32_t globalPhotonOffset;

    // 2026-05-17: GPU BEAM source 보조 (TOPAS Beam 정합)
    //   beamPosShape: 0=Rectangle, 1=Ellipse  (BeamPositionCutoffShape)
    //   beamPosSigmaX/Y: Gaussian sigma (mm). 0 → Flat (uniform within cutoff).
    //                                         >0 → Gaussian + cutoff clip (rejection).
    //   beamPolMode:  0=random transverse, 1=fixed (beamPolX/Y/Z 사용)
    //   beamPolX/Y/Z: fixed polarization (unit vector, beamPolMode=1 시)
    uint32_t beamPosShape;
    float    beamPosSigmaX;
    float    beamPosSigmaY;
    uint32_t beamPolMode;
    float    beamPolX, beamPolY, beamPolZ;

    // 2026-05-17 누락 항목 추가:
    //   beamPosDist:   0=None (point), 1=Flat, 2=Gaussian
    //   beamAngDist:   0=None (dir0), 1=Flat, 2=Gaussian, 3=Isotropic(4π, dir0 무시)
    //   beamEnergySpreadEv: energy Gaussian σ (eV); 0=mono
    //   beamTimeSpread:     time spread Gaussian σ (ns); 0=t=0 고정
    //   beamTimeCutoff:     time abs cutoff (ns); 0=no clip
    uint32_t beamPosDist;
    uint32_t beamAngDist;
    float    beamEnergySpreadEv;
    float    beamTimeSpread;
    float    beamTimeCutoff;

    // 2026-05-17 BeamEnergySpectrum (inline LUT, max 32 bins).
    //   beamSpectrumNumBins: 0=비활성 (mono+spread 만 사용)
    //   beamSpectrumType:    0=Discrete (단일 energy 선택), 1=Continuous (선형 interp)
    //   beamSpectrumEnergies[]:    bin energy (eV)
    //   beamSpectrumCumWeights[]:  cumulative normalized weights (0~1)
    // Discrete: 한 bin 의 단일 energy.
    // Continuous: 각 bin 의 [E_{j-1}, E_j] 안에서 cumW 의 inverse-CDF 선형 interp.
    uint32_t beamSpectrumNumBins;
    uint32_t beamSpectrumType;
    float    beamSpectrumEnergies[256];
    float    beamSpectrumCumWeights[256];
    // 2026-05-19: raw user weights (Continuous piecewise linear inverse CDF 의 PDF slope 계산)
    float    beamSpectrumWeights[256];
} MOPSimConfig;

// SF1 (2026-04-24): GPU surface flux scorer host structs (Common.h와 layout 동기화 필수)
// surface scorer 개수 제한 없음 — host std::vector + dynamic device buffer (CPU TOPAS 동일)

typedef struct MOP_ALIGNED(16) {
    float originX, originY, originZ;
    float normalX, normalY, normalZ;
    float axisUX, axisUY, axisUZ;
    float axisVX, axisVY, axisVZ;
    float halfExtentU;
    float halfExtentV;
    uint32_t surfaceId;
    uint32_t enabled;
} MOPSurfaceDef;

typedef struct MOP_ALIGNED(16) {
    float posX, posY, posZ;
    float dirX, dirY, dirZ;
    float energy;
    float cosTheta;        // signed: <0=going opposite to normal (in)
    float photonWeight;
    float time;
    uint32_t surfaceId;
    uint32_t flags;        // bit 0: 1=in, 0=out; bit 1: TIR; bit 2: s-pol
    uint32_t trackId;
    uint32_t materialIdFrom;
    uint32_t materialIdTo;
    uint32_t _pad;
} MOPSurfaceHit;

#ifndef __METAL_VERSION__
    #ifdef __cplusplus
    }
    #endif
#endif

#endif /* MOP_TYPES_H */
