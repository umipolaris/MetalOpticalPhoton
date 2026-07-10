/**
 * MetalOpticalEngine.mm
 * Objective-C++ 구현 - Metal GPU 엔진 코어
 *
 * M10 수정: GPU 버퍼 재사용 (매 Propagate 호출마다 재생성 방지)
 * CPU↔GPU 동기화 최적화: 비동기 완료 핸들러 + 더블 버퍼링
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "MetalOpticalEngine.h"
#include "../topas_extension/TopasParameterParser.hh"
#include <vector>
#include <iomanip>
#include <string>
#include <iostream>
#include <chrono>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <ios>
#include <dlfcn.h>

// ============================================================
// 내부 엔진 클래스
// ============================================================
class MetalOpticalEngineImpl {
public:
    MetalOpticalEngineImpl();
    ~MetalOpticalEngineImpl();

    bool Initialize();
    void SetConfig(const MOPSimConfig& config);
    MOPSimConfig GetConfig() const { return m_config; }

    // TOPAS 파라미터 로드
    int LoadTOPASParameters(const char* filePath);

    // 물질/표면 등록
    uint32_t RegisterMaterial(const MOPMaterialProperties& material, const char* name);
    // F2 정석 fix: material name → id 조회 (없으면 0xFFFFFFFF)
    uint32_t GetMaterialIdByName(const char* name) const {
        auto it = m_materialNames.find(std::string(name));
        return (it == m_materialNames.end()) ? 0xFFFFFFFFu : it->second;
    }
    uint32_t RegisterSurface(const MOPSurfaceProperties& surface, const char* name);

    // 지오메트리
    void AddVolumeMesh(const MOPTriangle* triangles, uint32_t numTriangles, uint32_t volumeId);
    void AddBoxGeometry(const MOPBoxGeometry& box);
    void AddSphereGeometry(const MOPSphereGeometry& sphere);
    void AddCylinderGeometry(const MOPCylinderGeometry& cyl);
    void AddTorusGeometry(const MOPTorusGeometry& tor);
    void AddPolyhedraGeometry(const MOPPolyhedraGeometry& ph);
    int BuildAccelerationStructure();
    void SetBorderSurface(uint32_t vol1, uint32_t vol2, uint32_t surfId);
    void SetSkinSurface(uint32_t vol, uint32_t surfId);

    // 시뮬레이션
    void AddGenstep(const MOPGenstep& genstep);
    // Primary opticalphoton (Beam source) 직접 주입 — gensteps 우회
    void AddPrimaryPhoton(const MOPPhoton& photon);
    void ResetPrimaryPhotons();
    uint32_t GetPrimaryPhotonCount() const;
    uint32_t Propagate();
    void WaitForGPU();  // Fix 71: 비동기 GPU 작업 완료 대기
    uint32_t GetHits(MOPHit* hits, uint32_t maxHits);
    void ResetGensteps();

    // Fix 41: Transit hit API
    uint32_t GetTransitHits(MOPHit* hits, uint32_t maxHits);
    uint32_t GetLastTransitCount() { WaitForGPU(); return m_lastTransitCount; }

    // SF1 (2026-04-24): GPU surface flux scorer API
    uint32_t RegisterSurfaceScorer(const MOPSurfaceDef& def);
    void ClearSurfaceScorers();
    uint32_t GetSurfaceHits(MOPSurfaceHit* hits, uint32_t maxHits);
    uint32_t GetLastSurfaceHitCount() const;

    // Fix 65: GPU DDA 스코어링
    void RunGPUDDA(float compTransX, float compTransY, float compTransZ,
                   float compFullX, float compFullY, float compFullZ,
                   uint32_t nBinsX, uint32_t nBinsY, uint32_t nBinsZ,
                   uint32_t scoringMaterialId = 0xFFFFFFFFu);
    const float* GetDDABinBuffer();  // 2026-05-12: uint32 → atomic_float (overflow fix)
    uint32_t GetDDABinCount() const { return m_ddaBinCount; }
    float GetDDAFluenceScale() const { return 1.0f; }  // 2026-05-12: float 누적이므로 scaling 불필요
    void WaitForDDA();  // Fix 78측정-2: 미완료 DDA cmdBuf 대기

    // Multi-scorer race fix (2026-05-01): per-scorer accumulating bin buffer.
    // 기존 RunGPUDDA 는 m_ddaBinBuffer 를 매 dispatch 마다 0-fill 후 누적 →
    // CPU 가 매 이벤트 read 해서 fEvtMap 으로 옮겨야 함 (CPU 동기화 부담).
    // 새 API: scorer 별 persistent buffer 를 등록하고 dispatch 마다 누적 →
    // CPU 는 EndOfRun 에서만 1회 read.
    int RegisterScorerBinBuffer(uint32_t totalBins, uint32_t nBinsE = 1,
                                float eMinEv = 0.0f, float eMaxEv = 0.0f);  // returns handle (>=0) or -1
    // 2026-05-19 multi-scorer: scorer 별 photon energy filter (eV). 0/0 = no filter (default).
    void SetScorerEnergyFilter(int handle, float energyLow, float energyHigh);
    void RunGPUDDA_Accumulate(int scorerHandle,
                              float compTransX, float compTransY, float compTransZ,
                              float compFullX, float compFullY, float compFullZ,
                              uint32_t nBinsX, uint32_t nBinsY, uint32_t nBinsZ,
                              uint32_t scoringMaterialId);
    // Phase 2.1+ (2026-05-16): cylindrical/spherical voxelization 지원 ext API.
    // voxelType: 0=BOX (Cartesian), 1=CYLINDER (R/Phi/Z), 2=SPHERE (R/Theta/Phi)
    // rMin: cyl/sph 의 inner radius (mm)
    // phiStart/thetaStart: cyl phi / sph theta 시작 각 (rad). default 0.
    void RunGPUDDA_AccumulateExt(int scorerHandle,
                                  float compTransX, float compTransY, float compTransZ,
                                  float compFullX, float compFullY, float compFullZ,
                                  uint32_t nBinsX, uint32_t nBinsY, uint32_t nBinsZ,
                                  uint32_t scoringMaterialId,
                                  uint32_t voxelType,
                                  float rMin, float phiStart, float thetaStart);
    const float* GetScorerBinBuffer(int scorerHandle, uint32_t* outCount);
    void ResetScorerBinBuffers();  // EndOfRun 후 다음 run 대비

    // Fix 41b: 스코어링 AABB 설정
    void SetScoringAABB(float minX, float minY, float minZ,
                        float maxX, float maxY, float maxZ) {
        // Bug A 후속 fix (2026-04-22): 다중 scorer 지원을 위해 union 으로
        // 변경. 기존 replace 방식은 마지막 등록된 AABB 만 사용 → 다른
        // scorer 의 region 은 transit 생성 안 됨. Union 으로 두면 kernel
        // 이 모든 scorer region 의 transit 을 생성하고, 각 scorer 의 DDA
        // 가 자신의 component bounds 로 clip 하여 정확한 path-length 추출.
        if (m_config.scoringAABBEnabled) {
            m_config.scoringAABBMinX = std::min(m_config.scoringAABBMinX, minX);
            m_config.scoringAABBMinY = std::min(m_config.scoringAABBMinY, minY);
            m_config.scoringAABBMinZ = std::min(m_config.scoringAABBMinZ, minZ);
            m_config.scoringAABBMaxX = std::max(m_config.scoringAABBMaxX, maxX);
            m_config.scoringAABBMaxY = std::max(m_config.scoringAABBMaxY, maxY);
            m_config.scoringAABBMaxZ = std::max(m_config.scoringAABBMaxZ, maxZ);
        } else {
            m_config.scoringAABBMinX = minX;
            m_config.scoringAABBMinY = minY;
            m_config.scoringAABBMinZ = minZ;
            m_config.scoringAABBMaxX = maxX;
            m_config.scoringAABBMaxY = maxY;
            m_config.scoringAABBMaxZ = maxZ;
            m_config.scoringAABBEnabled = 1;
        }
        if (m_logLevel >= 2) {
            printf("[MOP] Scoring AABB %s: (%.2f,%.2f,%.2f)-(%.2f,%.2f,%.2f) (union now: (%.2f,%.2f,%.2f)-(%.2f,%.2f,%.2f))\n",
                   "added",
                   minX, minY, minZ, maxX, maxY, maxZ,
                   m_config.scoringAABBMinX, m_config.scoringAABBMinY, m_config.scoringAABBMinZ,
                   m_config.scoringAABBMaxX, m_config.scoringAABBMaxY, m_config.scoringAABBMaxZ);
        }
    }

    // 상태
    bool IsGPUAvailable() const;
    void GetStats(MOPPropagationStats& stats) const;
    void SetLogLevel(int level);
    void SetBaseSeed(uint32_t seed) { m_baseSeed = seed; }  // 2026-05-12: TOPAS Ts/Seed
    void SetWorldSize(float x, float y, float z) {
        m_config.worldSizeX = x;
        m_config.worldSizeY = y;
        m_config.worldSizeZ = z;
    }

private:
    void CreateComputePipelines();
    void EnsurePropagatePipeline();  // Fix 78b: function_constant 기반 lazy 생성
    void UploadMaterialsToGPU();
    void UploadSurfacesToGPU();
    uint32_t CalculateTotalPhotons() const;
    void CalculateGenstepOffsets(std::vector<uint32_t>& offsets) const;

    // M10: 영속 GPU 버퍼 할당/크기 조정
    void EnsurePhotonBuffer(uint32_t count);
    void EnsureHitBuffer(uint32_t count);
    void EnsureGenstepBuffer(size_t count);

    // Metal 객체
    id<MTLDevice>              m_device;
    id<MTLCommandQueue>        m_commandQueue;
    id<MTLLibrary>             m_library;
    id<MTLComputePipelineState> m_generateScintPipeline;
    id<MTLComputePipelineState> m_generateCerenkovPipeline;
    id<MTLComputePipelineState> m_generateBeamPipeline;  // 2026-05-12: GPU beam source
    id<MTLComputePipelineState> m_propagatePipeline;

    // GPU 버퍼
    id<MTLBuffer> m_materialBuffer;
    id<MTLBuffer> m_surfaceBuffer;
    id<MTLBuffer> m_genstepBuffer;
    id<MTLBuffer> m_photonBuffer;
    id<MTLBuffer> m_photonMetaBuffer;  // Fix 78측정-6: cold meta (genstepId, parentTrackId)
    id<MTLBuffer> m_hitBuffer;
    id<MTLBuffer> m_hitCountBuffer;
    id<MTLBuffer> m_configBuffer;
    id<MTLBuffer> m_triangleBuffer;
    id<MTLBuffer> m_triangleCountBuffer;
    id<MTLBuffer> m_boxGeometryBuffer;
    id<MTLBuffer> m_boxGeometryCountBuffer;
    std::vector<MOPBoxGeometry> m_boxGeometries;

    // Phase 1 (2026-05-05): G4Sphere analytic intersection (in-kernel loop).
    // Box (G1) 와 동일 패턴: BVH 우회, kernel 안에서 ray-sphere quadratic 직접 풀이.
    // Custom Intersection Function Table 인프라 안 씀 (Box 와 일관성 유지).
    id<MTLBuffer> m_sphereGeometryBuffer;
    id<MTLBuffer> m_sphereGeometryCountBuffer;
    std::vector<MOPSphereGeometry> m_sphereGeometries;

    // Phase 2 (2026-05-05): G4Tubs analytic intersection (in-kernel loop).
    id<MTLBuffer> m_cylinderGeometryBuffer;
    id<MTLBuffer> m_cylinderGeometryCountBuffer;
    std::vector<MOPCylinderGeometry> m_cylinderGeometries;

    // Phase 3 (2026-05-05): G4Torus analytic intersection (Ferrari quartic).
    id<MTLBuffer> m_torusGeometryBuffer;
    id<MTLBuffer> m_torusGeometryCountBuffer;
    std::vector<MOPTorusGeometry> m_torusGeometries;

    // Phase 4 (2026-05-18): G4Polyhedra (general N-sided) analytic intersection.
    id<MTLBuffer> m_polyhedraGeometryBuffer;
    id<MTLBuffer> m_polyhedraGeometryCountBuffer;
    std::vector<MOPPolyhedraGeometry> m_polyhedraGeometries;
    id<MTLBuffer> m_genstepOffsetBuffer;
    id<MTLBuffer> m_photonCountBuffer;
    id<MTLBuffer> m_batchCountBuffer;  // M10: 영속 배치 카운트 버퍼

    // Fix 41: Transit hit 버퍼 (스코어링 볼륨 통과 기록)
    id<MTLBuffer> m_transitHitBuffer;
    id<MTLBuffer> m_transitCountBuffer;
    id<MTLBuffer> m_maxTransitHitsBuffer;
    uint32_t m_transitBufferCapacity;
    bool m_transitGrowRequested = false;  // 2026-05-03: overflow detect 시 next dispatch 에서 2x grow
    uint32_t m_lastTransitCount;

    // SF1 (2026-04-24): GPU surface flux scorer 버퍼
    // 개수 제한 없음 — std::vector + dynamic device buffer (CPU TOPAS 동일)
    id<MTLBuffer> m_surfaceDefBuffer;     // MOPSurfaceDef[m_surfaceDefCapacity]
    id<MTLBuffer> m_surfaceHitBuffer;     // MOPSurfaceHit[capacity]
    id<MTLBuffer> m_surfaceHitCountBuffer; // atomic uint32
    id<MTLBuffer> m_maxSurfaceHitsBuffer;  // uint32 (capacity)
    uint32_t m_surfaceHitCapacity;
    uint32_t m_lastSurfaceHitCount;
    std::vector<MOPSurfaceDef> m_surfaceDefs;
    uint32_t m_surfaceDefCapacity;        // m_surfaceDefBuffer 가 담을 수 있는 def 개수
    uint32_t m_numActiveSurfaces;

    // 진단: 경계면 이벤트 카운터 버퍼 (32 x uint32_t, dc[0]-dc[31])
    id<MTLBuffer> m_diagBuffer;

    // Fix 61: 궤적 덤프 버퍼
    id<MTLBuffer> m_trajectoryBuffer;
    id<MTLBuffer> m_trajectoryCountBuffer;

    // Fix 64: 재질 LUT 버퍼
    id<MTLBuffer> m_materialLUTBuffer;
    id<MTLBuffer> m_surfaceLUTBuffer;   // Fix 71: 표면 LUT

    // Fix 65: GPU DDA 스코어링
    id<MTLComputePipelineState> m_ddaPipeline;
    id<MTLBuffer> m_ddaBinBuffer;      // atomic uint32 빈 배열
    id<MTLCommandBuffer> m_pendingDDACmdBuf;  // Fix 78측정-2: 비동기 DDA 추적
    id<MTLBuffer> m_ddaConfigBuffer;   // DDAConfig
    uint32_t m_ddaBinCount;            // 현재 빈 수
    bool m_ddaEnabled;                 // GPU DDA 활성화 여부
    // 2026-05-15: fp32 ULP saturation 감지용 1-element atomic_float (lostValueSum).
    //   DDAScoring.metal 이 contrib 가 ULP round-lost 됐을 때 lost 값 자체를 누적.
    //   host (GetDDABinBuffer / GetScorerBinBuffer) 가 dispatch 후 read,
    //   binBuffer 총합 대비 비율 > 1% 면 abort. 참조: docs/gpu_dda_fp32_saturation.md
    id<MTLBuffer> m_saturationFlagBuffer;

    // Multi-scorer race fix (2026-05-01 / 2026-05-02 lean):
    // per-scorer 모든 GPU state 분리 → multi-scorer 동시 dispatch 시 race 원천 차단.
    //   - bin buffer (accumulating)
    //   - DDA config buffer (CPU writes per dispatch — 공유면 race)
    //   - pending DDA cmdBuf (per-scorer 독립 sync)
    struct ScorerBinBuffer {
        id<MTLBuffer> buffer;             // 누적 빈 버퍼 (2026-05-17: uint64 emulation, totalBins × 2 × uint32)
        id<MTLBuffer> configBuffer;       // per-scorer DDAConfig (race 차단)
        id<MTLCommandBuffer> pendingCmdBuf;  // per-scorer DDA pending (독립 sync)
        uint32_t totalBins;
        std::vector<float> reconstructedView;  // 2026-05-17: GetScorerBinBuffer 가 uint64 → float / SCALE 변환 후 cache
        // 2026-05-19 multi-scorer: photon energy filter (eV). 0/0 = no filter (all accept).
        float energyLow = 0.0f;
        float energyHigh = 0.0f;
        // 2026-05-20: energy/wavelength binning (CPU Fluence EBins 동등). nBinsE=1 = 기존.
        // buffer 는 totalBins(voxel) × nBinsE 개. layout [eBin*totalBins + voxel].
        uint32_t nBinsE = 1;
        float eMinEv = 0.0f;
        float eMaxEv = 0.0f;
    };
    std::vector<ScorerBinBuffer> m_scorerBuffers;

    // Fix 71: 비동기 GPU 파이프라이닝
    id<MTLCommandBuffer> m_pendingCmdBuf;   // 미완료 command buffer
    bool m_hasPendingWork;                   // 대기 중인 GPU 작업 유무
    uint32_t m_pendingTotalPhotons;          // 대기 중인 배치의 광자 수
    std::chrono::high_resolution_clock::time_point m_pendingSubmitTime; // commit 시각

    // M10: 현재 할당된 버퍼 용량 추적
    uint32_t m_photonBufferCapacity;
    uint32_t m_hitBufferCapacity;
    size_t   m_genstepBufferCapacity;
    size_t   m_genstepOffsetCapacity;

    // CPU 측 데이터
    MOPSimConfig                    m_config;
    std::vector<MOPMaterialProperties> m_materials;
    std::vector<MOPSurfaceProperties>  m_surfaces;
    std::vector<MOPTriangle>           m_triangles;

    // Lock-free genstep 수집: 원자적 카운터 + 사전할당 배열
    static const uint32_t GENSTEP_CAPACITY = 1048576;  // 1M gensteps (Fix 68: 대형 배치)
    MOPGenstep* m_genstepStorage;
    std::atomic<uint32_t> m_genstepCount{0};

    // Primary opticalphoton 직접 주입용 storage (Beam source path)
    static const uint32_t PRIMARY_PHOTON_CAPACITY = 1048576;  // 1M primary photons
    MOPPhoton* m_primaryPhotonStorage = nullptr;
    std::atomic<uint32_t> m_primaryPhotonCount{0};

    // 히트: GPU 버퍼 직접 접근 (복사 제거)
    uint32_t m_lastHitCount;

    // 이름 맵핑 (TOPAS 이름 → ID)
    std::unordered_map<std::string, uint32_t> m_materialNames;
    std::unordered_map<std::string, uint32_t> m_surfaceNames;

    // TOPAS 파라미터 파서
    TopasParameterParser m_parser;

    // 통계
    MOPPropagationStats m_stats;
    int m_logLevel;
    bool m_initialized;
    uint32_t m_eventCounter;        // 이벤트별 시드 변화용
    uint32_t m_baseSeed;            // 2026-05-12: TOPAS Ts/Seed (mixed per batch)
    bool m_accelerationStructureBuilt;

    // Metal Ray Tracing 가속 구조
    id<MTLAccelerationStructure> m_accelStructure;
    id<MTLBuffer> m_vertexPositionBuffer;

    // Two-BVH (Geant4 parallel-world 패턴) — 별도 scoring-only AS.
    // mass-BVH (m_accelStructure) 와 독립적으로 query하여 trivial scoring
    // 볼륨(Air-in-Air sensor 등) 의 phantom-hit 이 광학 boundary 처리를
    //방해하지 않도록 함. Phase 1: 인프라만 (1개 dummy triangle), 미사용.
    id<MTLAccelerationStructure> m_scoringAccelStructure;
    id<MTLBuffer> m_scoringVertexPositionBuffer;
    bool m_scoringAccelStructureBuilt;

};

// ============================================================
// 구현
// ============================================================
MetalOpticalEngineImpl::MetalOpticalEngineImpl()
    : m_device(nil)
    , m_commandQueue(nil)
    , m_library(nil)
    , m_logLevel(1)
    , m_initialized(false)
    , m_accelerationStructureBuilt(false)
    , m_accelStructure(nil)
    , m_vertexPositionBuffer(nil)
    , m_scoringAccelStructure(nil)
    , m_scoringVertexPositionBuffer(nil)
    , m_scoringAccelStructureBuilt(false)
    , m_photonBufferCapacity(0)
    , m_hitBufferCapacity(0)
    , m_genstepBufferCapacity(0)
    , m_genstepOffsetCapacity(0)
    , m_batchCountBuffer(nil)
    , m_hitCountBuffer(nil)
    , m_photonCountBuffer(nil)
    , m_configBuffer(nil)
    , m_genstepStorage(nullptr)
    , m_genstepCount(0)
    , m_lastHitCount(0)
    , m_transitHitBuffer(nil)
    , m_transitCountBuffer(nil)
    , m_maxTransitHitsBuffer(nil)
    , m_transitBufferCapacity(0)
    , m_lastTransitCount(0)
    // SF1 (2026-04-24): GPU surface flux scorer
    , m_surfaceDefBuffer(nil)
    , m_surfaceHitBuffer(nil)
    , m_surfaceHitCountBuffer(nil)
    , m_maxSurfaceHitsBuffer(nil)
    , m_surfaceHitCapacity(0)
    , m_lastSurfaceHitCount(0)
    , m_surfaceDefCapacity(0)
    , m_numActiveSurfaces(0)
    , m_pendingCmdBuf(nil)
    , m_hasPendingWork(false)
    , m_pendingTotalPhotons(0)
    , m_sphereGeometryBuffer(nil)
    , m_sphereGeometryCountBuffer(nil)
    , m_cylinderGeometryBuffer(nil)
    , m_cylinderGeometryCountBuffer(nil)
    , m_torusGeometryBuffer(nil)
    , m_torusGeometryCountBuffer(nil)
    , m_polyhedraGeometryBuffer(nil)
    , m_polyhedraGeometryCountBuffer(nil)
{
    memset(&m_config, 0, sizeof(m_config));
    m_surfaceDefs.clear();             // SF1: surface scorer 비활성 시작
    m_config.numSurfaces = 0;
    // SF1 deficit 검증: CPU TOPAS Ts/MaxStepNumber default = 1,000,000
    // 이전 GPU default 10,000은 100배 적음 → 1% systematic T deficit 후보
    m_config.maxStepsPerPhoton = 1000000;  // CPU TOPAS Ts/MaxStepNumber default = 1,000,000 맞춤 (이전 1000 은 1000× 부족 → WLS reemit 등 TIR 갇힌 광자가 MAX_STEPS kill 되어 plate fluence 누락)
    m_config.maxPhotonsPerBatch = MOP_MAX_PHOTONS_PER_BATCH;
    m_config.randomSeed = 42;
    m_baseSeed = 42;  // 2026-05-12: TOPAS Ts/Seed mix base. Set via SetBaseSeed.
    m_config.globalPhotonOffset = 0;  // 2026-05-12 fix: batch-cumulative photon index
    m_eventCounter = 0;
    m_config.enableScintillation = 1;
    m_config.enableCerenkov = 1;
    m_config.enableAbsorption = 1;
    m_config.enableRayleigh = 1;
    m_config.enableMie = 1;
    m_config.enableWLS = 1;
    m_config.enableBoundary = 1;
    m_config.worldSizeX = 10000.0f;
    m_config.worldSizeY = 10000.0f;
    m_config.worldSizeZ = 10000.0f;
    m_config.cerenkovMaxPhotonsPerStep = 300;

    m_config.cerenkovMaxBetaChange = 10.0f;
    m_config.numGensteps = 0;

    memset(&m_stats, 0, sizeof(m_stats));

    // surfaceId=0을 "표면 없음"으로 예약: 더미 표면 삽입
    MOPSurfaceProperties dummySurf;
    memset(&dummySurf, 0, sizeof(dummySurf));
    m_surfaces.push_back(dummySurf);
}

MetalOpticalEngineImpl::~MetalOpticalEngineImpl() {
    // Fix 63: 마지막 pending GPU 작업 대기
    if (m_hasPendingWork && m_pendingCmdBuf) {
        [m_pendingCmdBuf waitUntilCompleted];
        m_pendingCmdBuf = nil;
        m_hasPendingWork = false;
    }
    // Fix 78측정-2: 미완료 DDA cmdBuf 안전 대기
    if (m_pendingDDACmdBuf) {
        [m_pendingDDACmdBuf waitUntilCompleted];
        m_pendingDDACmdBuf = nil;
    }
    delete[] m_genstepStorage;
    delete[] m_primaryPhotonStorage;
    // ARC handles Metal object release
}

bool MetalOpticalEngineImpl::Initialize() {
    @autoreleasepool {
        m_device = MTLCreateSystemDefaultDevice();
        if (!m_device) {
            if (m_logLevel >= 1)
                std::cerr << "[MOP] Error: No Metal GPU device found" << std::endl;
            return false;
        }

        if (m_logLevel >= 3) {
            std::cout << "[MOP] Metal device: "
                      << [[m_device name] UTF8String] << std::endl;
            std::cout << "[MOP] Unified memory: "
                      << ([m_device hasUnifiedMemory] ? "Yes" : "No") << std::endl;
        }

        m_commandQueue = [m_device newCommandQueue];
        if (!m_commandQueue) {
            std::cerr << "[MOP] Error: Failed to create command queue" << std::endl;
            return false;
        }

        // Metal 셰이더 라이브러리 로드
        NSError* error = nil;

        // 1. 환경변수 MOP_METALLIB_PATH
        const char* envPath = getenv("MOP_METALLIB_PATH");
        if (envPath) {
            NSString* libPath = [NSString stringWithUTF8String:envPath];
            NSURL* libURL = [NSURL fileURLWithPath:libPath];
            m_library = [m_device newLibraryWithURL:libURL error:&error];
            if (m_library && m_logLevel >= 2)
                std::cout << "[MOP] Loaded metallib from MOP_METALLIB_PATH: " << envPath << std::endl;
        }

        // 2. dylib 디렉토리
        if (!m_library) {
            Dl_info dlInfo;
            static int _mop_anchor = 0;
            if (dladdr((void*)&_mop_anchor, &dlInfo)) {
                NSString* dylibPath = [NSString stringWithUTF8String:dlInfo.dli_fname];
                NSString* dylibDir = [dylibPath stringByDeletingLastPathComponent];
                NSString* metallibPath = [dylibDir stringByAppendingPathComponent:@"default.metallib"];
                NSURL* libURL = [NSURL fileURLWithPath:metallibPath];
                m_library = [m_device newLibraryWithURL:libURL error:&error];
                if (m_library && m_logLevel >= 2)
                    std::cout << "[MOP] Loaded metallib from dylib directory: "
                              << [metallibPath UTF8String] << std::endl;
            }
        }

        // 3. 현재 작업 디렉토리
        if (!m_library) {
            NSString* cwdPath = [[NSFileManager defaultManager] currentDirectoryPath];
            NSString* metallibPath = [cwdPath stringByAppendingPathComponent:@"default.metallib"];
            NSURL* libURL = [NSURL fileURLWithPath:metallibPath];
            m_library = [m_device newLibraryWithURL:libURL error:&error];
            if (m_library && m_logLevel >= 2)
                std::cout << "[MOP] Loaded metallib from CWD: "
                          << [metallibPath UTF8String] << std::endl;
        }

        // 4. 기본 라이브러리
        if (!m_library) {
            m_library = [m_device newDefaultLibrary];
        }

        if (!m_library) {
            std::cerr << "[MOP] ERROR: Could not load Metal shader library (default.metallib). "
                      << "Set MOP_METALLIB_PATH or place it next to libMetalOpticalPhoton.dylib." << std::endl;
            if (error)
                std::cerr << "[MOP] Metal error: " << [[error localizedDescription] UTF8String] << std::endl;
        }

        CreateComputePipelines();

        // M10: 영속 보조 버퍼 초기 할당
        uint32_t zero = 0;
        m_hitCountBuffer = [m_device newBufferWithBytes:&zero length:sizeof(uint32_t)
                                     options:MTLResourceStorageModeShared];
        m_photonCountBuffer = [m_device newBufferWithBytes:&zero length:sizeof(uint32_t)
                                        options:MTLResourceStorageModeShared];
        m_batchCountBuffer = [m_device newBufferWithLength:sizeof(uint32_t)
                                       options:MTLResourceStorageModeShared];

        // Fix 41: Transit hit 보조 버퍼 초기 할당
        m_transitCountBuffer = [m_device newBufferWithBytes:&zero length:sizeof(uint32_t)
                                         options:MTLResourceStorageModeShared];
        uint32_t defaultMaxTransit = 500000;
        m_maxTransitHitsBuffer = [m_device newBufferWithBytes:&defaultMaxTransit length:sizeof(uint32_t)
                                           options:MTLResourceStorageModeShared];

        // SF1 (2026-04-24): GPU surface flux scorer def 버퍼
        // 개수 제한 없음 — RegisterSurfaceScorer 에서 필요 시 동적 재할당.
        // dispatch 시 항상 atIndex:23 으로 바인딩되므로 nil 방지용 dummy 1개로 초기화.
        m_surfaceDefCapacity = 1;
        m_surfaceDefBuffer = [m_device newBufferWithLength:m_surfaceDefCapacity * sizeof(MOPSurfaceDef)
                                          options:MTLResourceStorageModeShared];
        memset([m_surfaceDefBuffer contents], 0, m_surfaceDefCapacity * sizeof(MOPSurfaceDef));
        m_surfaceHitCountBuffer = [m_device newBufferWithBytes:&zero length:sizeof(uint32_t)
                                              options:MTLResourceStorageModeShared];
        uint32_t defaultMaxSurfaceHits = 500000;  // 5명 합의: 500k SurfaceHits 최대
        m_maxSurfaceHitsBuffer = [m_device newBufferWithBytes:&defaultMaxSurfaceHits length:sizeof(uint32_t)
                                             options:MTLResourceStorageModeShared];

        m_configBuffer = [m_device newBufferWithBytes:&m_config length:sizeof(MOPSimConfig)
                                    options:MTLResourceStorageModeShared];

        // 진단: 경계면 이벤트 카운터 버퍼 (64 x uint32_t)
        // CPU TOPAS 비교용 reflect counters: [32]=FresnelTrueReflect, [33]=FresnelTrueRefract,
        // [34]=FresnelTrueTIR, [35]=SameMaterialEarlyExit, [36]=WithSurfaceTrueReflect,
        // [37]=WithSurfaceTrueRefract, [38]=WithSurfaceTrueTIR, [39]=WithSurfaceTransmission
        // Expert reflect_dir (2026-04-22): buffer 64→128 확장 (slot 80-87 사용)
        // Expert SideX-hit-pos (2026-04-22): buffer 128→2048 확장
        //   slot 100         = SideX TIR hit dump 카운터 (0..100)
        //   slot 200..1099   = (x,y,z,dirx,diry,dirz,nx,ny,nz) 9 uint32 raw float per entry
        //                       100 entries × 9 = 900 slots
        m_diagBuffer = [m_device newBufferWithLength:8192 * sizeof(uint32_t)
                                   options:MTLResourceStorageModeShared];

        // Fix 61: 궤적 덤프 버퍼 — Common.h MAX_TRAJECTORY_STEPS=40000 와 일치
        m_trajectoryBuffer = [m_device newBufferWithLength:500000 * sizeof(MOPTrajectoryStep)
                                       options:MTLResourceStorageModeShared];
        m_trajectoryCountBuffer = [m_device newBufferWithLength:sizeof(uint32_t)
                                            options:MTLResourceStorageModeShared];

        // Lock-free genstep 배열 사전 할당
        m_genstepStorage = new MOPGenstep[GENSTEP_CAPACITY];
        // Primary opticalphoton 배열 사전 할당 (Beam source path)
        m_primaryPhotonStorage = new MOPPhoton[PRIMARY_PHOTON_CAPACITY];

        m_initialized = true;
        if (m_logLevel >= 3)
            std::cout << "[MOP] Engine initialized successfully" << std::endl;

        return true;
    }
}

// Fix 78b: propagate 파이프라인을 function_constant로 빌드 (enableXxx dead branch 제거)
void MetalOpticalEngineImpl::EnsurePropagatePipeline() {
    if (m_propagatePipeline) return;
    if (!m_library) return;

    @autoreleasepool {
        NSError* error = nil;

        // function_constant 값 준비 — 현재 config의 enable* 플래그를 컴파일 타임 상수로
        MTLFunctionConstantValues* constants = [[MTLFunctionConstantValues alloc] init];
        bool fcAbs  = (m_config.enableAbsorption != 0);
        bool fcRay  = (m_config.enableRayleigh   != 0);
        bool fcMie  = (m_config.enableMie        != 0);
        bool fcWLS  = (m_config.enableWLS        != 0);
        bool fcBnd  = (m_config.enableBoundary   != 0);
        [constants setConstantValue:&fcAbs  type:MTLDataTypeBool atIndex:0];
        [constants setConstantValue:&fcRay  type:MTLDataTypeBool atIndex:1];
        [constants setConstantValue:&fcMie  type:MTLDataTypeBool atIndex:2];
        [constants setConstantValue:&fcWLS  type:MTLDataTypeBool atIndex:3];
        [constants setConstantValue:&fcBnd  type:MTLDataTypeBool atIndex:4];

        id<MTLFunction> propagateFunc = [m_library newFunctionWithName:@"propagatePhotons"
                                                        constantValues:constants
                                                                 error:&error];
        if (!propagateFunc) {
            std::cerr << "[MOP] propagate function specialization failed: "
                      << (error ? [[error description] UTF8String] : "unknown") << std::endl;
            return;
        }
        m_propagatePipeline = [m_device newComputePipelineStateWithFunction:propagateFunc error:&error];
        if (!m_propagatePipeline) {
            std::cerr << "[MOP] propagate pipeline creation failed: "
                      << (error ? [[error description] UTF8String] : "unknown") << std::endl;
            return;
        }
        if (m_logLevel >= 2) {
            std::cout << "[MOP] Propagation pipeline (function_constant): abs=" << fcAbs
                      << " ray=" << fcRay << " mie=" << fcMie << " wls=" << fcWLS
                      << " bnd=" << fcBnd
                      << " maxTPT=" << [m_propagatePipeline maxTotalThreadsPerThreadgroup]
                      << std::endl;
        }
    }
}

void MetalOpticalEngineImpl::CreateComputePipelines() {
    if (!m_library) return;

    @autoreleasepool {
        NSError* error = nil;

        id<MTLFunction> scintFunc = [m_library newFunctionWithName:@"generateScintillationPhotons"];
        if (scintFunc) {
            m_generateScintPipeline = [m_device newComputePipelineStateWithFunction:scintFunc error:&error];
        }

        id<MTLFunction> cerenkovFunc = [m_library newFunctionWithName:@"generateCerenkovPhotons"];
        if (cerenkovFunc) {
            m_generateCerenkovPipeline = [m_device newComputePipelineStateWithFunction:cerenkovFunc error:&error];
        }

        // 2026-05-12: GPU beam photon generation kernel
        id<MTLFunction> beamFunc = [m_library newFunctionWithName:@"generateBeamPhotons"];
        if (beamFunc) {
            m_generateBeamPipeline = [m_device newComputePipelineStateWithFunction:beamFunc error:&error];
            if (m_logLevel >= 2)
                std::cout << "[MOP] Beam photon generation pipeline created" << std::endl;
        }

        // Fix 78b: propagate 파이프라인은 function_constant 바인딩을 위해
        // 첫 Propagate() 호출 시 lazy 생성 (EnsurePropagatePipeline)
        m_propagatePipeline = nil;

        // Fix 65: GPU DDA 파이프라인
        id<MTLFunction> ddaFunc = [m_library newFunctionWithName:@"ddaScoring"];
        if (ddaFunc) {
            m_ddaPipeline = [m_device newComputePipelineStateWithFunction:ddaFunc error:&error];
            if (m_logLevel >= 2)
                std::cout << "[MOP] DDA scoring pipeline created" << std::endl;
        }
        m_ddaBinBuffer = nil;
        m_ddaConfigBuffer = [m_device newBufferWithLength:sizeof(MOPDDAConfig)
                                      options:MTLResourceStorageModeShared];
        m_ddaBinCount = 0;
        m_ddaEnabled = false;
        // 2026-05-15: ULP saturation lost-value 누적 (1 atomic_float, shared 메모리).
        m_saturationFlagBuffer = [m_device newBufferWithLength:sizeof(float)
                                            options:MTLResourceStorageModeShared];
        *(float*)[m_saturationFlagBuffer contents] = 0.0f;
    }
}

void MetalOpticalEngineImpl::SetConfig(const MOPSimConfig& config) {
    m_config = config;
}

// ============================================================
// M10: 영속 버퍼 관리 — 필요 시만 재할당 (grow-only)
// ============================================================
void MetalOpticalEngineImpl::EnsurePhotonBuffer(uint32_t count) {
    if (count <= m_photonBufferCapacity && m_photonBuffer) return;
    // 25% 여유 용량으로 할당 (빈번한 재할당 방지)
    uint32_t newCapacity = count + count / 4;
    // Fix 78측정-4: untracked hazard mode — 우리가 WaitForGPU/WaitForDDA로 명시 동기화
    m_photonBuffer = [m_device newBufferWithLength:(NSUInteger)newCapacity * sizeof(MOPPhoton)
                               options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked];
    // Fix 78측정-6: cold meta 버퍼 함께 할당
    m_photonMetaBuffer = [m_device newBufferWithLength:(NSUInteger)newCapacity * sizeof(MOPPhotonMeta)
                                   options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked];
    m_photonBufferCapacity = newCapacity;
    if (m_logLevel >= 3)
        std::cout << "[MOP] Photon buffer resized: " << newCapacity << " photons ("
                  << (newCapacity * sizeof(MOPPhoton) / 1048576) << " MB)" << std::endl;
}

void MetalOpticalEngineImpl::EnsureHitBuffer(uint32_t count) {
    if (count <= m_hitBufferCapacity && m_hitBuffer) return;
    uint32_t newCapacity = count + count / 4;
    m_hitBuffer = [m_device newBufferWithLength:newCapacity * sizeof(MOPHit)
                            options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked];
    m_hitBufferCapacity = newCapacity;
}

void MetalOpticalEngineImpl::EnsureGenstepBuffer(size_t count) {
    if (count <= m_genstepBufferCapacity && m_genstepBuffer) return;
    size_t newCapacity = std::max((size_t)1, count + count / 4);  // Fix 73: 최소 1
    m_genstepBuffer = [m_device newBufferWithLength:newCapacity * sizeof(MOPGenstep)
                                options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked];
    m_genstepBufferCapacity = newCapacity;

    // 오프셋 버퍼도 함께 확장
    size_t offsetCapacity = newCapacity + 1;
    m_genstepOffsetBuffer = [m_device newBufferWithLength:offsetCapacity * sizeof(uint32_t)
                                      options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked];
    m_genstepOffsetCapacity = offsetCapacity;
}

// ============================================================
// TOPAS 파라미터 로드
// ============================================================
int MetalOpticalEngineImpl::LoadTOPASParameters(const char* filePath) {
    int ret = m_parser.ParseFile(std::string(filePath));
    if (ret != 0) return ret;

    for (const auto& mat : m_parser.GetMaterials()) {
        RegisterMaterial(mat.props, mat.name.c_str());
    }

    for (const auto& surf : m_parser.GetSurfaces()) {
        RegisterSurface(surf.props, surf.name.c_str());
    }

    m_config.enableScintillation = m_parser.IsScintillationEnabled() ? 1 : 0;
    m_config.enableCerenkov = m_parser.IsCerenkovEnabled() ? 1 : 0;
    m_config.enableAbsorption = m_parser.IsOpticalAbsorptionEnabled() ? 1 : 0;
    m_config.enableRayleigh = m_parser.IsRayleighEnabled() ? 1 : 0;
    m_config.enableMie = m_parser.IsMieEnabled() ? 1 : 0;
    m_config.enableWLS = m_parser.IsWLSEnabled() ? 1 : 0;
    m_config.enableBoundary = m_parser.IsBoundaryEnabled() ? 1 : 0;
    m_config.cerenkovMaxPhotonsPerStep = m_parser.GetCerenkovMaxPhotonsPerStep();
    m_config.cerenkovMaxBetaChange = m_parser.GetCerenkovMaxBetaChange();

    if (m_logLevel >= 3) {
        std::cout << "[MOP] TOPAS parameters loaded: "
                  << m_materials.size() << " materials, "
                  << m_surfaces.size() << " surfaces" << std::endl;
    }

    return 0;
}

uint32_t MetalOpticalEngineImpl::RegisterMaterial(
    const MOPMaterialProperties& material, const char* name)
{
    uint32_t id = (uint32_t)m_materials.size();
    MOPMaterialProperties mat = material;
    mat.materialId = id;
    m_materials.push_back(mat);
    m_materialNames[std::string(name)] = id;
    return id;
}

uint32_t MetalOpticalEngineImpl::RegisterSurface(
    const MOPSurfaceProperties& surface, const char* name)
{
    uint32_t id = (uint32_t)m_surfaces.size();
    MOPSurfaceProperties surf = surface;
    surf.surfaceId = id;
    m_surfaces.push_back(surf);
    m_surfaceNames[std::string(name)] = id;
    return id;
}

// ============================================================
// 지오메트리
// ============================================================
void MetalOpticalEngineImpl::AddVolumeMesh(
    const MOPTriangle* triangles, uint32_t numTriangles, uint32_t volumeId)
{
    for (uint32_t i = 0; i < numTriangles; i++) {
        m_triangles.push_back(triangles[i]);
    }
}

int MetalOpticalEngineImpl::BuildAccelerationStructure() {
    @autoreleasepool {
        // 2026-05-15: materials/surfaces 는 triangle 유무와 무관하게 항상 GPU 로 upload.
        //   scintillation 광자 생성 + propagate kernel 이 materials/surfaces buffer 를
        //   사용하므로, box-analytic-only 셋업 (triangle 0) 에서도 필수.
        //   box analytic 재활성화(8ff2e68) 후 box-only 셋업이 early return 으로 빠져
        //   m_materialBuffer 가 nil → "missing Buffer binding for materials[0]" 였음.
        UploadMaterialsToGPU();
        UploadSurfacesToGPU();
        // 2026-05-15 (정석 fix): box-analytic-only 셋업 (triangle 0개) 을 정상 상태로 지원.
        //   box analytic 재활성화(8ff2e68) 전엔 box 가 mesh 라 triangle 이 항상 ≥1 →
        //   "triangle ≥1, BVH valid" invariant 위에 코드가 작성됐음. analytic 전환 후
        //   mesh 없는 셋업 (U5_short 등) 이 early return → GPU sum 0 였음.
        //   이전엔 가짜(degenerate) triangle 1개를 주입해 invariant 를 속였으나, 그건
        //   꼼수 — triCount 를 1 로 거짓말함. 정석: triCount=0 을 진실로 두고 아래
        //   각 단계가 triCount==0 을 정상 분기로 처리한다:
        //     (1) BVH centroid: triCount==0 이면 계산 skip (0 으로 나눗셈 회피)
        //     (2) vertex / TriangleAttr 버퍼: triCount==0 이면 binding 용 1-element
        //         placeholder 버퍼 (kernel 은 triCount=0 이라 절대 index 안 함, 또
        //         intersector 가 빈 BVH 라 hit 자체가 안 나옴 → physics 무영향)
        //     (3) 가속구조: triangleCount=0 인 빈 BVH 빌드 → intersector 가 항상 miss.
        if (m_triangles.empty() && m_logLevel >= 2)
            std::cerr << "[MOP] No mesh triangles — analytic-geometry-only setup (empty BVH)" << std::endl;

        // Fix 71: TriangleAttr 32B 경량 버퍼 생성 (기존 Triangle 64B 대비 50% 축소)
        // 2026-05-10 H7: vertex 추가 (12*3=36 byte) → 68 byte. brute-force triangle
        // intersect on GPU (BVH 우회) 위한 shader-side vertex access.
        {
            // 2026-05-15: H7 vertex (v0/v1/v2) 제거 — shader read 0회 dead, 68B→32B.
            struct TriangleAttrCPU {
                float normal[3];
                uint32_t volumeId;
                uint32_t materialIdInside;
                uint32_t materialIdOutside;
                uint32_t surfaceId;
                uint32_t _pad;
            };
            // triCount==0 (analytic-only): binding 용 1-element placeholder.
            //   value-init 으로 0 채워짐, kernel 은 triCount=0 이라 read 안 함.
            size_t triAttrN = m_triangles.empty() ? 1 : m_triangles.size();
            std::vector<TriangleAttrCPU> triAttrs(triAttrN);
            for (size_t i = 0; i < m_triangles.size(); i++) {
                triAttrs[i].normal[0] = m_triangles[i].nx;
                triAttrs[i].normal[1] = m_triangles[i].ny;
                triAttrs[i].normal[2] = m_triangles[i].nz;
                triAttrs[i].volumeId = m_triangles[i].volumeId;
                triAttrs[i].materialIdInside = m_triangles[i].materialIdInside;
                triAttrs[i].materialIdOutside = m_triangles[i].materialIdOutside;
                triAttrs[i].surfaceId = m_triangles[i].surfaceId;
                triAttrs[i]._pad = 0;
            }
            size_t triAttrSize = triAttrs.size() * sizeof(TriangleAttrCPU);
            m_triangleBuffer = [m_device newBufferWithBytes:triAttrs.data()
                                         length:triAttrSize
                                         options:MTLResourceStorageModeShared];
            if (m_logLevel >= 1) {
                std::cout << "[MOP-Fix71] TriangleAttr buffer: " << triAttrs.size()
                          << " x 32B = " << (triAttrSize / 1024.0 / 1024.0) << " MB"
                          << " (was " << (m_triangles.size() * sizeof(MOPTriangle) / 1024.0 / 1024.0) << " MB)" << std::endl;
            }
        }

        uint32_t triCount = (uint32_t)m_triangles.size();
        m_triangleCountBuffer = [m_device newBufferWithBytes:&triCount
                                          length:sizeof(uint32_t)
                                          options:MTLResourceStorageModeShared];

        // G1: G4Box analytic geometry buffer
        {
            uint32_t boxCount = (uint32_t)m_boxGeometries.size();
            if (boxCount == 0) {
                MOPBoxGeometry dummy = {};
                m_boxGeometryBuffer = [m_device newBufferWithBytes:&dummy
                                            length:sizeof(MOPBoxGeometry)
                                            options:MTLResourceStorageModeShared];
            } else {
                m_boxGeometryBuffer = [m_device newBufferWithBytes:m_boxGeometries.data()
                                            length:boxCount * sizeof(MOPBoxGeometry)
                                            options:MTLResourceStorageModeShared];
            }
            m_boxGeometryCountBuffer = [m_device newBufferWithBytes:&boxCount
                                            length:sizeof(uint32_t)
                                            options:MTLResourceStorageModeShared];
            if (m_logLevel >= 2) std::cout << "[MOP-G1] BoxGeometry buffer: " << boxCount << " entries\n";
        }

        // Phase 1: G4Sphere analytic geometry buffer (in-kernel loop)
        uint32_t sphereCount = (uint32_t)m_sphereGeometries.size();
        {
            if (sphereCount == 0) {
                MOPSphereGeometry dummy = {};
                m_sphereGeometryBuffer = [m_device newBufferWithBytes:&dummy
                                            length:sizeof(MOPSphereGeometry)
                                            options:MTLResourceStorageModeShared];
            } else {
                m_sphereGeometryBuffer = [m_device newBufferWithBytes:m_sphereGeometries.data()
                                            length:sphereCount * sizeof(MOPSphereGeometry)
                                            options:MTLResourceStorageModeShared];
            }
            m_sphereGeometryCountBuffer = [m_device newBufferWithBytes:&sphereCount
                                            length:sizeof(uint32_t)
                                            options:MTLResourceStorageModeShared];
            if (m_logLevel >= 2) std::cout << "[MOP-Sphere] SphereGeometry buffer: " << sphereCount << " entries\n";
        }

        // Phase 2: G4Tubs analytic geometry buffer (in-kernel loop)
        // Phase 3 와 count 통합: buffer 30 = uint2{numCyl, numTor}
        uint32_t cylCount = (uint32_t)m_cylinderGeometries.size();
        uint32_t torCount = (uint32_t)m_torusGeometries.size();
        {
            if (cylCount == 0) {
                MOPCylinderGeometry dummy = {};
                m_cylinderGeometryBuffer = [m_device newBufferWithBytes:&dummy
                                            length:sizeof(MOPCylinderGeometry)
                                            options:MTLResourceStorageModeShared];
            } else {
                m_cylinderGeometryBuffer = [m_device newBufferWithBytes:m_cylinderGeometries.data()
                                            length:cylCount * sizeof(MOPCylinderGeometry)
                                            options:MTLResourceStorageModeShared];
            }
            // 통합 count buffer (uint4 = cylCount, torCount, polyCount, sphereCount)
            // Phase 4 (2026-05-18): polyhedra + sphere count 도 같은 buffer 에 packing
            uint32_t allCounts[4] = { cylCount, torCount, 0, sphereCount };  // polyCount filled later
            m_cylinderGeometryCountBuffer = [m_device newBufferWithBytes:allCounts
                                            length:sizeof(allCounts)
                                            options:MTLResourceStorageModeShared];
            if (m_logLevel >= 2) std::cout << "[MOP-Cyl] CylinderGeometry buffer: " << cylCount << " entries\n";
        }

        // Phase 3: G4Torus analytic geometry buffer (Ferrari quartic)
        {
            if (torCount == 0) {
                MOPTorusGeometry dummy = {};
                m_torusGeometryBuffer = [m_device newBufferWithBytes:&dummy
                                            length:sizeof(MOPTorusGeometry)
                                            options:MTLResourceStorageModeShared];
            } else {
                m_torusGeometryBuffer = [m_device newBufferWithBytes:m_torusGeometries.data()
                                            length:torCount * sizeof(MOPTorusGeometry)
                                            options:MTLResourceStorageModeShared];
            }
            // count 는 m_cylinderGeometryCountBuffer 의 두 번째 uint 에 packing
            m_torusGeometryCountBuffer = nil;  // 미사용 (slot 확보 위해)
            if (m_logLevel >= 2) std::cout << "[MOP-Torus] TorusGeometry buffer: " << torCount << " entries\n";
        }

        // Phase 4 (2026-05-18): G4Polyhedra analytic geometry buffer
        uint32_t polyCount = (uint32_t)m_polyhedraGeometries.size();
        {
            if (polyCount == 0) {
                MOPPolyhedraGeometry dummy = {};
                m_polyhedraGeometryBuffer = [m_device newBufferWithBytes:&dummy
                                            length:sizeof(MOPPolyhedraGeometry)
                                            options:MTLResourceStorageModeShared];
            } else {
                m_polyhedraGeometryBuffer = [m_device newBufferWithBytes:m_polyhedraGeometries.data()
                                            length:polyCount * sizeof(MOPPolyhedraGeometry)
                                            options:MTLResourceStorageModeShared];
            }
            m_polyhedraGeometryCountBuffer = nil;  // count 는 cylinder buffer 의 uint4[2] 에 packing
            // Update existing cylinder count buffer with polyCount
            uint32_t* counts = (uint32_t*)[m_cylinderGeometryCountBuffer contents];
            counts[2] = polyCount;
            if (m_logLevel >= 2) std::cout << "[MOP-Poly] PolyhedraGeometry buffer: " << polyCount << " entries\n";
        }

        // (UploadMaterialsToGPU / UploadSurfacesToGPU 는 함수 시작부로 이동 — 위 주석 참조)

        // === Metal Hardware Ray Tracing 가속 구조 빌드 ===
        // HW RT precision 실험 (2026-04-22) 결과:
        //   - vertex stride 16B padded vs 12B tight: 동일 결과 (영향 없음)
        //   - usage flag Refit vs None: BVH topology 다름 → 결과 변화하지만
        //     개선이 아닌 statistical noise. None(default)이 가장 안정.
        //   Apple Metal HW RT는 사용자가 조정할 수 있는 precision/watertight
        //   옵션을 노출하지 않음 (intersector flag, geometry usage flag 모두
        //   precision 제어 불가). 따라서 기본 설정 유지.
        uint32_t numVertices = triCount * 3;
        // triCount==0 (analytic-only): vertex buffer 는 non-nil 이어야 binding 가능.
        //   빈 BVH 라 intersector 가 hit 자체를 안 내므로 내용은 무의미 — 9 floats
        //   (triangle 1개분) 0-fill placeholder.
        std::vector<float> vertexData(numVertices > 0 ? numVertices * 3 : 9, 0.0f);
        // 2026-05-08 fp32 precision fix: BVH centroid offset.
        // All mesh vertices shifted by -centroid before BVH build → fp32 ULP at
        // shifted coords ≈ ULP near zero, much smaller than at original world coords.
        // 광자 ray cast 시 photon.position - centroid 적용 (kernel 측). 결과 distance
        // 는 translation invariant 이므로 정확. world 변환 자동 (photon += dir*dist).
        double cx = 0, cy = 0, cz = 0;
        for (uint32_t i = 0; i < triCount; i++) {
            const MOPTriangle& tri = m_triangles[i];
            cx += (double)tri.v0x + (double)tri.v1x + (double)tri.v2x;
            cy += (double)tri.v0y + (double)tri.v1y + (double)tri.v2y;
            cz += (double)tri.v0z + (double)tri.v1z + (double)tri.v2z;
        }
        // triCount==0 이면 centroid 정의 불가 → 0 (mesh 없으니 shift 도 불필요).
        double invN = (triCount > 0) ? 1.0 / (3.0 * (double)triCount) : 0.0;
        m_config.bvhCentroidX = (float)(cx * invN);
        m_config.bvhCentroidY = (float)(cy * invN);
        m_config.bvhCentroidZ = (float)(cz * invN);
        if (m_logLevel >= 1) {
            std::cout << "[MOP-BVHCentroid] BVH centroid offset: ("
                      << m_config.bvhCentroidX << ", "
                      << m_config.bvhCentroidY << ", "
                      << m_config.bvhCentroidZ << ") mm — "
                      << "vertex coords shifted for fp32 precision" << std::endl;
        }
        for (uint32_t i = 0; i < triCount; i++) {
            const MOPTriangle& tri = m_triangles[i];
            vertexData[i * 9 + 0] = tri.v0x - m_config.bvhCentroidX;
            vertexData[i * 9 + 1] = tri.v0y - m_config.bvhCentroidY;
            vertexData[i * 9 + 2] = tri.v0z - m_config.bvhCentroidZ;
            vertexData[i * 9 + 3] = tri.v1x - m_config.bvhCentroidX;
            vertexData[i * 9 + 4] = tri.v1y - m_config.bvhCentroidY;
            vertexData[i * 9 + 5] = tri.v1z - m_config.bvhCentroidZ;
            vertexData[i * 9 + 6] = tri.v2x - m_config.bvhCentroidX;
            vertexData[i * 9 + 7] = tri.v2y - m_config.bvhCentroidY;
            vertexData[i * 9 + 8] = tri.v2z - m_config.bvhCentroidZ;
        }
        m_vertexPositionBuffer = [m_device newBufferWithBytes:vertexData.data()
                                            length:vertexData.size() * sizeof(float)
                                            options:MTLResourceStorageModeShared];

        MTLAccelerationStructureTriangleGeometryDescriptor* geomDesc =
            [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
        geomDesc.vertexBuffer = m_vertexPositionBuffer;
        geomDesc.vertexBufferOffset = 0;
        geomDesc.vertexStride = 3 * sizeof(float);
        geomDesc.vertexFormat = MTLAttributeFormatFloat3;
        geomDesc.triangleCount = triCount;
        // Fix 78d: 모든 삼각형을 opaque로 선언 — intersection function 경로 제거
        //          intersector의 force_opacity(opaque)와 함께 BVH leaf traversal 가속
        geomDesc.opaque = YES;

        MTLPrimitiveAccelerationStructureDescriptor* accelDesc =
            [MTLPrimitiveAccelerationStructureDescriptor descriptor];
        accelDesc.geometryDescriptors = @[ geomDesc ];
        // Fix 78d: usage None = 기본값이자 fast intersection 최적화 (명시)
        accelDesc.usage = MTLAccelerationStructureUsageNone;

        MTLAccelerationStructureSizes sizes = [m_device accelerationStructureSizesWithDescriptor:accelDesc];
        // triangleCount=0 (analytic-only) 일 때도 Metal 은 valid 한 (작은) AS size 를
        //   반환함 — intersector 가 항상 intersection_type::none 을 내는 빈 BVH.
        //   kernel 의 accelStruct binding (buffer 9) 은 non-nil 이어야 하므로 size 가
        //   0 으로 나오는 예외 케이스만 1 로 보정해 non-nil AS 를 보장한다.
        NSUInteger asSize = sizes.accelerationStructureSize > 0 ? sizes.accelerationStructureSize : 1;
        m_accelStructure = [m_device newAccelerationStructureWithSize:asSize];
        id<MTLBuffer> scratchBuffer = [m_device newBufferWithLength:(sizes.buildScratchBufferSize > 0 ? sizes.buildScratchBufferSize : 1)
                                                 options:MTLResourceStorageModePrivate];

        id<MTLCommandBuffer> buildCmdBuf = [m_commandQueue commandBuffer];
        id<MTLAccelerationStructureCommandEncoder> buildEncoder =
            [buildCmdBuf accelerationStructureCommandEncoder];
        [buildEncoder buildAccelerationStructure:m_accelStructure
                                     descriptor:accelDesc
                                  scratchBuffer:scratchBuffer
                            scratchBufferOffset:0];
        [buildEncoder endEncoding];
        [buildCmdBuf commit];
        [buildCmdBuf waitUntilCompleted];

        m_accelerationStructureBuilt = true;

        if (m_logLevel >= 2) {
            std::cout << "[MOP] Metal RT acceleration structure built: "
                      << triCount << " triangles"
                      << (triCount == 0 ? " (empty BVH — analytic-only setup)" : "")
                      << ", " << (sizes.accelerationStructureSize / 1024) << " KB" << std::endl;
        }

        // ============================================================
        // Two-BVH Phase 1: scoring-only AS 빌드 (현재는 1개 dummy triangle).
        //
        // Geant4 parallel-world 패턴 이식: trivial scoring face (Air-in-Air
        // sensor 등) 를 mass-BVH에서 분리해 별도 AS로 관리. 광자 propagation
        // 의 boundary 결정은 mass-BVH로만 하고, scoring transit 검출은 이
        // scoring-BVH로만 함. Phase 1에서는 인프라 + 1개 dummy triangle 만
        // 빌드해 kernel binding pipeline을 검증. trivial face 분류는 Phase 2.
        // ============================================================
        {
            // dummy: World 밖 멀리 떨어진 triangle 1개 (실제 hit 발생 안 함)
            float dummyVerts[9] = {
                1e6f, 1e6f, 1e6f,
                1e6f + 0.001f, 1e6f, 1e6f,
                1e6f, 1e6f + 0.001f, 1e6f
            };
            m_scoringVertexPositionBuffer =
                [m_device newBufferWithBytes:dummyVerts
                                      length:sizeof(dummyVerts)
                                     options:MTLResourceStorageModeShared];

            MTLAccelerationStructureTriangleGeometryDescriptor* sgeomDesc =
                [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
            sgeomDesc.vertexBuffer = m_scoringVertexPositionBuffer;
            sgeomDesc.vertexBufferOffset = 0;
            sgeomDesc.vertexStride = 3 * sizeof(float);
            sgeomDesc.vertexFormat = MTLAttributeFormatFloat3;
            sgeomDesc.triangleCount = 1;
            sgeomDesc.opaque = YES;

            MTLPrimitiveAccelerationStructureDescriptor* sAccelDesc =
                [MTLPrimitiveAccelerationStructureDescriptor descriptor];
            sAccelDesc.geometryDescriptors = @[ sgeomDesc ];
            sAccelDesc.usage = MTLAccelerationStructureUsageNone;

            MTLAccelerationStructureSizes ssizes =
                [m_device accelerationStructureSizesWithDescriptor:sAccelDesc];
            m_scoringAccelStructure =
                [m_device newAccelerationStructureWithSize:ssizes.accelerationStructureSize];
            id<MTLBuffer> sScratch =
                [m_device newBufferWithLength:ssizes.buildScratchBufferSize
                                      options:MTLResourceStorageModePrivate];

            id<MTLCommandBuffer> sBuildCmdBuf = [m_commandQueue commandBuffer];
            id<MTLAccelerationStructureCommandEncoder> sBuildEnc =
                [sBuildCmdBuf accelerationStructureCommandEncoder];
            [sBuildEnc buildAccelerationStructure:m_scoringAccelStructure
                                       descriptor:sAccelDesc
                                    scratchBuffer:sScratch
                              scratchBufferOffset:0];
            [sBuildEnc endEncoding];
            [sBuildCmdBuf commit];
            [sBuildCmdBuf waitUntilCompleted];

            m_scoringAccelStructureBuilt = true;

            if (m_logLevel >= 2) {
                std::cout << "[MOP] Two-BVH Phase 1: scoring AS (dummy) built — "
                          << (ssizes.accelerationStructureSize / 1024) << " KB"
                          << std::endl;
            }
        }

        // Fix 64: 재질 LUT 생성 — PropertyTable을 64엔트리 균일 격자로 변환
        {
            uint32_t matCount = (uint32_t)m_materials.size();
            std::vector<MOPMaterialLUT> luts(MOP_MAX_MATERIALS);
            memset(luts.data(), 0, sizeof(MOPMaterialLUT) * MOP_MAX_MATERIALS);

            // 2026-05-20: natural cubic spline interpolation (CPU TOPAS applySpline=true 와 동일).
            // TsMaterialManager.cc:300 이 모든 vector property (RIndex 포함) 에 G4 spline 적용 →
            // CPU 는 RIndex 를 cubic spline 으로 매끄럽게. 이전 GPU linear interpolation 은
            // 11-point RIndex 의 kink 를 LUT 에 보존 → dispersion fluence 계단 (z-profile step).
            // 2026-06-09 정정: 이전엔 natural cubic spline 을 썼으나 G4 는 not-a-knot
            // (ComputeSecDerivative1) 이라 끝점 구간에서 갈렸다. 아래에서 G4 와 동일한
            // not-a-knot 으로 채운다 → GPU n(λ) == CPU n(λ) (band edge 분산차 제거).
            auto fillLUT = [](float* lutOut, const MOPPropertyTable& table) {
                const float eMin = 1.0f;  // PHOTON_ENERGY_MIN
                const float eMax = 15.0f; // PHOTON_ENERGY_MAX
                uint32_t n = table.count;
                if (n == 0) { for (uint32_t i = 0; i < MOP_LUT_SIZE; i++) lutOut[i] = 0.0f; return; }
                if (n == 1) { for (uint32_t i = 0; i < MOP_LUT_SIZE; i++) lutOut[i] = table.values[0]; return; }
                // 2026-06-09: G4 not-a-knot cubic spline second derivatives (y2), matching
                // Geant4/TOPAS. CPU TOPAS (TsMaterialManager applySpline=true) interpolates
                // RIndex with G4PhysicsVector G4SplineType::Base = ComputeSecDerivative1
                // (not-a-knot). The previous *natural* spline here (y2[0]=y2[n-1]=0) used the
                // wrong endpoint boundary condition: natural and not-a-knot diverge in the
                // first/last table interval, shifting n(λ) by ~3e-4 at the spectrum edge that
                // falls in that interval (e.g. SF1 red end 1.65 eV is inside the first interval
                // [1.5,1.8]). Through the dispersion cascade that n(λ) offset produced a
                // wavelength-dependent deflection difference of order ~0.5 mm at the red band
                // edge vs CPU. not-a-knot here makes GPU n(λ) identical to Geant4.
                const float* xt = table.energies;
                const float* yt = table.values;
                std::vector<double> y2(n, 0.0);
                if (n >= 5) {  // G4 Base nmin=5; fewer points → y2=0 (linear), same as G4
                    uint32_t nn = n - 1;
                    std::vector<double> u(n, 0.0);
                    u[1] = (double)(yt[2]-yt[1])/(xt[2]-xt[1]) - (double)(yt[1]-yt[0])/(xt[1]-xt[0]);
                    u[1] = 6.0*u[1]*(xt[2]-xt[1])/((double)(xt[2]-xt[0])*(xt[2]-xt[0]));
                    y2[1] = (2.0*xt[1]-xt[0]-xt[2])/(2.0*xt[2]-xt[0]-xt[1]);
                    for (uint32_t i = 2; i < nn - 1; i++) {
                        double sig = (double)(xt[i]-xt[i-1])/(xt[i+1]-xt[i-1]);
                        double p = sig*y2[i-1] + 2.0;
                        y2[i] = (sig - 1.0)/p;
                        u[i] = (double)(yt[i+1]-yt[i])/(xt[i+1]-xt[i]) - (double)(yt[i]-yt[i-1])/(xt[i]-xt[i-1]);
                        u[i] = 6.0*u[i]/(xt[i+1]-xt[i-1]) - sig*u[i-1]/p;
                    }
                    double sig = (double)(xt[nn-1]-xt[nn-2])/(xt[nn]-xt[nn-2]);
                    double p = sig*y2[nn-3] + 2.0;
                    u[nn-1] = (double)(yt[nn]-yt[nn-1])/(xt[nn]-xt[nn-1]) - (double)(yt[nn-1]-yt[nn-2])/(xt[nn-1]-xt[nn-2]);
                    u[nn-1] = 6.0*sig*u[nn-1]/(xt[nn]-xt[nn-2]) - (2.0*sig-1.0)*u[nn-2]/p;
                    p = (1.0+sig) + (2.0*sig-1.0)*y2[nn-2];
                    y2[nn-1] = u[nn-1]/p;
                    for (int k = (int)nn - 2; k > 1; k--)
                        y2[k] *= (y2[k+1] - u[k]*(double)(xt[k+1]-xt[k-1])/(xt[k+1]-xt[k]));
                    y2[nn] = (y2[nn-1] - (1.0-sig)*y2[nn-2])/sig;
                    sig = 1.0 - ((double)(xt[2]-xt[1])/(xt[2]-xt[0]));
                    y2[1] *= (y2[2] - u[1]/(1.0-sig));
                    y2[0] = (y2[1] - sig*y2[2])/(1.0-sig);
                }
                for (uint32_t i = 0; i < MOP_LUT_SIZE; i++) {
                    float energy = eMin + (eMax - eMin) * float(i) / float(MOP_LUT_SIZE - 1);
                    if (energy <= table.energies[0]) { lutOut[i] = table.values[0]; continue; }
                    if (energy >= table.energies[n-1]) { lutOut[i] = table.values[n-1]; continue; }
                    uint32_t lo = 0, hi = n - 1;
                    while (hi - lo > 1) { uint32_t mid = (lo+hi)/2; if (table.energies[mid] <= energy) lo = mid; else hi = mid; }
                    double h = table.energies[hi] - table.energies[lo];
                    double a = (table.energies[hi] - energy) / h;
                    double b = (energy - table.energies[lo]) / h;
                    lutOut[i] = (float)(a * table.values[lo] + b * table.values[hi]
                              + ((a*a*a - a) * y2[lo] + (b*b*b - b) * y2[hi]) * h * h / 6.0);
                }
            };

            // 2026-05-20: length property (흡수/산란 길이) 는 linear interpolation.
            // RINDEX 는 dispersion (매끄러운 n(λ)) 이라 cubic spline 이 맞지만, length
            // property 는 step·큰 dynamic range (예: WLSABSLENGTH 1e8→1mm) 라 cubic spline
            // 이 overshoot → 비물리적 음수 (wlsLen<0 → shader if(wlsLen>0) WLS skip bug).
            //
            // 2026-05-24 fix: absLen 을 G4 와 동일한 not-a-knot cubic spline 으로 (fillLUTSplineG4).
            // CPU TOPAS (TsMaterialManager.cc:300 applySpline=true) 는 absLen 도 spline
            // (G4PhysicsVector G4SplineType::Base = ComputeSecDerivative1, not-a-knot). GPU linear 과
            // coarse·가파른 파장의존 abslen (07 물 6점 3cm→5000cm) 에서 보간이 갈라져 긴경로(확산)
            // 광자 생존 차이 → on-axis↔확산 재분배 (07 G/C ~1.10). G4 와 동일 spline 으로 매칭.
            // G4 의 spline 은 가파른 구간서 음수로 overshoot (예 물 abslen 4.2-5eV) → G4 는 음수 MFP →
            // 음수 GPIL → 즉시 흡수. GPU 는 음수 LUT 를 tiny-positive(ABSLEN_MIN) 로 clamp → 즉시 흡수
            // 근사 (shader distToAbsorb≈0). naive natural cubic spline(fillLUT) 직접 사용은 음수 →
            // shader if(absLen>0) skip → 무한 trapping → overflow 라 불가. rayLen/mie/wls 는 linear 유지.
            auto fillLUTLinear = [](float* lutOut, const MOPPropertyTable& table) {
                const float eMin = 1.0f, eMax = 15.0f;
                uint32_t n = table.count;
                if (n == 0) { for (uint32_t i = 0; i < MOP_LUT_SIZE; i++) lutOut[i] = 0.0f; return; }
                if (n == 1) { for (uint32_t i = 0; i < MOP_LUT_SIZE; i++) lutOut[i] = table.values[0]; return; }
                for (uint32_t i = 0; i < MOP_LUT_SIZE; i++) {
                    float energy = eMin + (eMax - eMin) * float(i) / float(MOP_LUT_SIZE - 1);
                    if (energy <= table.energies[0]) { lutOut[i] = table.values[0]; continue; }
                    if (energy >= table.energies[n-1]) { lutOut[i] = table.values[n-1]; continue; }
                    uint32_t lo = 0, hi = n - 1;
                    while (hi - lo > 1) { uint32_t mid = (lo+hi)/2; if (table.energies[mid] <= energy) lo = mid; else hi = mid; }
                    float t = (energy - table.energies[lo]) / (table.energies[hi] - table.energies[lo]);
                    lutOut[i] = table.values[lo] * (1.0f - t) + table.values[hi] * t;
                }
            };

            // 2026-05-24: absLen per-photon 정확 eval 용 G4 not-a-knot spline knot 을 absLen[256] 에 패킹.
            // (struct 변경 회피 — Metal struct stride 이슈 우회.) layout: [0]=N, [1..64]=E, [65..128]=V,
            // [129..192]=D2 (1+3*64=193 <= 256). secD = G4 ComputeSecDerivative1 (G4PhysicsVector.cc:290-352).
            // shader SplineEvalG4Packed 가 광자 에너지에서 직접 cubic eval (LUT 보간오차 0).
            auto fillAbsLenPacked = [](float* out, const MOPPropertyTable& table) {
                for (uint32_t i = 0; i < MOP_LUT_SIZE; i++) out[i] = 0.0f;
                uint32_t N = table.count;
                if (N == 0 || N > MOP_MAX_KNOTS) { out[0] = 0.0f; return; }  // 없음/초과 → shader no-absorb
                const float* x = table.energies;
                const float* y = table.values;
                std::vector<double> secD(N, 0.0);
                if (N >= 5) {  // G4 Base nmin=5; <5 면 secD=0 (linear)
                    uint32_t n = N - 1;
                    std::vector<double> u(N, 0.0);
                    u[1] = (double)(y[2]-y[1])/(x[2]-x[1]) - (double)(y[1]-y[0])/(x[1]-x[0]);
                    u[1] = 6.0*u[1]*(x[2]-x[1])/((double)(x[2]-x[0])*(x[2]-x[0]));
                    secD[1] = (2.0*x[1]-x[0]-x[2])/(2.0*x[2]-x[0]-x[1]);
                    for (uint32_t i = 2; i < n - 1; i++) {
                        double sig = (double)(x[i]-x[i-1])/(x[i+1]-x[i-1]);
                        double p = sig*secD[i-1] + 2.0;
                        secD[i] = (sig - 1.0)/p;
                        u[i] = (double)(y[i+1]-y[i])/(x[i+1]-x[i]) - (double)(y[i]-y[i-1])/(x[i]-x[i-1]);
                        u[i] = 6.0*u[i]/(x[i+1]-x[i-1]) - sig*u[i-1]/p;
                    }
                    double sig = (double)(x[n-1]-x[n-2])/(x[n]-x[n-2]);
                    double p = sig*secD[n-3] + 2.0;
                    u[n-1] = (double)(y[n]-y[n-1])/(x[n]-x[n-1]) - (double)(y[n-1]-y[n-2])/(x[n-1]-x[n-2]);
                    u[n-1] = 6.0*sig*u[n-1]/(x[n]-x[n-2]) - (2.0*sig-1.0)*u[n-2]/p;
                    p = (1.0+sig) + (2.0*sig-1.0)*secD[n-2];
                    secD[n-1] = u[n-1]/p;
                    for (int k = (int)n - 2; k > 1; k--)
                        secD[k] *= (secD[k+1] - u[k]*(double)(x[k+1]-x[k-1])/(x[k+1]-x[k]));
                    secD[n] = (secD[n-1] - (1.0-sig)*secD[n-2])/sig;
                    sig = 1.0 - ((double)(x[2]-x[1])/(x[2]-x[0]));
                    secD[1] *= (secD[2] - u[1]/(1.0-sig));
                    secD[0] = (secD[1] - sig*secD[2])/(1.0-sig);
                }
                out[0] = (float)N;
                for (uint32_t i = 0; i < N; i++) {
                    out[1 + i]   = x[i];          // E
                    out[65 + i]  = y[i];          // V
                    out[129 + i] = (float)secD[i]; // D2
                }
            };

            for (uint32_t m = 0; m < matCount && m < MOP_MAX_MATERIALS; m++) {
                const auto& mat = m_materials[m];
                fillLUT(luts[m].rindex, mat.refractiveIndex);        // cubic spline (dispersion)
                fillAbsLenPacked(luts[m].absLen, mat.absorptionLength); // per-photon 정확 eval (packed knots)
                fillLUTLinear(luts[m].rayLen, mat.rayleighLength);
                fillLUTLinear(luts[m].mieLen, mat.mieScattering);
                // 2026-06-04 fix: wlsLen 도 absLen 처럼 packed raw 점 (per-photon 직접 eval).
                //   256점 LUT 는 WLSABSLENGTH 의 가파른 절벽(1e8→2.0mm)을 뭉개 CsI 발광 2.25eV 서
                //   투명값 → GPU under-absorb (입사광 과다침투, plate 재방출 과소). CPU 는 raw 점 직접
                //   평가(절벽 보존). fillAbsLenPacked: N<5 면 D2=0 → linear-on-raw (G4 CPU 와 동일).
                fillAbsLenPacked(luts[m].wlsLen, mat.wlsAbsLength);
            }

            m_materialLUTBuffer = [m_device newBufferWithBytes:luts.data()
                                            length:sizeof(MOPMaterialLUT) * MOP_MAX_MATERIALS
                                            options:MTLResourceStorageModeShared];
            // 2026-05-19 debug: dump LUT rindex 일부 entries (low energy region) — gated (LogLevel>=2)
            if (m_logLevel >= 2) {
            std::cout << "[LUT-DEBUG] Material count=" << matCount << std::endl;
            for (uint32_t m = 0; m < std::min(matCount, (uint32_t)5); m++) {
                const auto& mat = m_materials[m];
                std::cout << "[LUT-DEBUG] mat[" << m << "] RIndex table.count=" << mat.refractiveIndex.count
                          << " values=[";
                for (uint32_t k = 0; k < std::min((uint32_t)mat.refractiveIndex.count, (uint32_t)3); k++)
                    std::cout << mat.refractiveIndex.values[k] << ",";
                std::cout << "...] energies=[";
                for (uint32_t k = 0; k < std::min((uint32_t)mat.refractiveIndex.count, (uint32_t)3); k++)
                    std::cout << mat.refractiveIndex.energies[k] << ",";
                std::cout << "...]" << std::endl;
                std::cout << "[LUT-DEBUG]   LUT[0..30] rindex (E=1.0..2.65 eV): ";
                for (uint32_t i = 0; i < 31; i++)
                    std::cout << luts[m].rindex[i] << ",";
                std::cout << std::endl;
                std::cout << "[LUT-DEBUG]   absLen.count=" << mat.absorptionLength.count
                          << " values=[";
                for (uint32_t k = 0; k < std::min((uint32_t)mat.absorptionLength.count, (uint32_t)3); k++)
                    std::cout << mat.absorptionLength.values[k] << ",";
                std::cout << "...] LUT[18]=" << luts[m].absLen[18] << " (idx E=2.0 absLen)" << std::endl;
            }
            }  // end LUT-DEBUG gate (LogLevel>=2)
            if (m_logLevel >= 2) {
                std::cout << "[MOP] Fix 64: Material LUT buffer created ("
                          << matCount << " materials × " << MOP_LUT_SIZE << " entries)"
                          << std::endl;
            }
        }

        // Fix 71: Surface LUT 생성
        {
            std::vector<MOPSurfaceLUT> surfLUTs(MOP_MAX_SURFACES);
            memset(surfLUTs.data(), 0, sizeof(MOPSurfaceLUT) * MOP_MAX_SURFACES);

            // fillLUT과 동일한 로직 재사용 (eMin=1.0, eMax=15.0)
            auto fillSurfLUT = [](float* lut, const MOPPropertyTable& table) {
                const float eMin = 1.0f, eMax = 15.0f;
                for (uint32_t i = 0; i < MOP_LUT_SIZE; i++) {
                    float energy = eMin + (eMax - eMin) * float(i) / float(MOP_LUT_SIZE - 1);
                    if (table.count == 0) { lut[i] = 0.0f; continue; }
                    if (table.count == 1) { lut[i] = table.values[0]; continue; }
                    if (energy <= table.energies[0]) { lut[i] = table.values[0]; continue; }
                    if (energy >= table.energies[table.count-1]) { lut[i] = table.values[table.count-1]; continue; }
                    uint32_t lo = 0, hi = table.count - 1;
                    while (hi - lo > 1) { uint32_t mid = (lo+hi)/2; if (table.energies[mid] <= energy) lo = mid; else hi = mid; }
                    float e0 = table.energies[lo], e1 = table.energies[hi];
                    float dE = e1 - e0;
                    float t = (dE < 1e-20f) ? 0.5f : (energy - e0) / dE;
                    lut[i] = table.values[lo] + t * (table.values[hi] - table.values[lo]);
                }
            };

            uint32_t surfCount = 0;
            for (uint32_t s = 0; s < MOP_MAX_SURFACES && s < m_surfaces.size(); s++) {
                fillSurfLUT(surfLUTs[s].reflectivity, m_surfaces[s].reflectivity);
                fillSurfLUT(surfLUTs[s].efficiency, m_surfaces[s].efficiency);
                fillSurfLUT(surfLUTs[s].transmittance, m_surfaces[s].transmittance);
                fillSurfLUT(surfLUTs[s].specularLobe, m_surfaces[s].specularLobe);
                fillSurfLUT(surfLUTs[s].specularSpike, m_surfaces[s].specularSpike);
                fillSurfLUT(surfLUTs[s].backScatter, m_surfaces[s].backScatter);
                surfCount++;
            }

            m_surfaceLUTBuffer = [m_device newBufferWithBytes:surfLUTs.data()
                                            length:sizeof(MOPSurfaceLUT) * MOP_MAX_SURFACES
                                            options:MTLResourceStorageModeShared];
            if (m_logLevel >= 1) {
                std::cout << "[MOP-Fix71] Surface LUT buffer: " << surfCount
                          << " surfaces × 6 props × " << MOP_LUT_SIZE << " entries = "
                          << (sizeof(MOPSurfaceLUT) * MOP_MAX_SURFACES / 1024.0) << " KB" << std::endl;
            }
        }

        // 디버그: 렌즈 삼각형 vertex 데이터 검증
        if (m_logLevel >= 3 && triCount > 100) {
            // 첫 12개 삼각형(World) 이후의 삼각형 (렌즈 등)에서 몇 개 덤프
            for (uint32_t i = 12; i < std::min(triCount, (uint32_t)20); i++) {
                const MOPTriangle& tri = m_triangles[i];
                float vx0 = vertexData[i*9+0], vy0 = vertexData[i*9+1], vz0 = vertexData[i*9+2];
                float vx1 = vertexData[i*9+3], vy1 = vertexData[i*9+4], vz1 = vertexData[i*9+5];
                float vx2 = vertexData[i*9+6], vy2 = vertexData[i*9+7], vz2 = vertexData[i*9+8];
                std::cout << "[MOP]   Tri[" << i << "] vtxBuf: v0=(" << vx0 << "," << vy0 << "," << vz0
                          << ") v1=(" << vx1 << "," << vy1 << "," << vz1
                          << ") v2=(" << vx2 << "," << vy2 << "," << vz2
                          << ") tri: v0=(" << tri.v0x << "," << tri.v0y << "," << tri.v0z
                          << ") n=(" << tri.nx << "," << tri.ny << "," << tri.nz
                          << ") volId=" << tri.volumeId << " matIn=" << tri.materialIdInside
                          << std::endl;
            }
        }

        return 0;
    }
}

void MetalOpticalEngineImpl::UploadMaterialsToGPU() {
    if (m_materials.empty()) {
        MOPMaterialProperties dummy;
        memset(&dummy, 0, sizeof(dummy));
        m_materialBuffer = [m_device newBufferWithBytes:&dummy
                                    length:sizeof(MOPMaterialProperties)
                                    options:MTLResourceStorageModeShared];
        return;
    }
    size_t size = m_materials.size() * sizeof(MOPMaterialProperties);
    m_materialBuffer = [m_device newBufferWithBytes:m_materials.data()
                                 length:size
                                 options:MTLResourceStorageModeShared];
}

void MetalOpticalEngineImpl::UploadSurfacesToGPU() {
    if (m_surfaces.empty()) {
        MOPSurfaceProperties dummy;
        memset(&dummy, 0, sizeof(dummy));
        m_surfaceBuffer = [m_device newBufferWithBytes:&dummy
                                    length:sizeof(MOPSurfaceProperties)
                                    options:MTLResourceStorageModeShared];
        return;
    }
    if (m_logLevel >= 2) {
        for (size_t i = 0; i < m_surfaces.size(); i++) {
            const auto& s = m_surfaces[i];
            std::cout << "[MOP] Surface[" << i << "] id=" << s.surfaceId
                      << " type=" << s.type << " finish=" << s.finish
                      << " model=" << s.model
                      << " forceR=" << s.forceReflectivity
                      << " reflLUT.count=" << s.reflectivity.count
                      << std::endl;
        }
    }
    size_t size = m_surfaces.size() * sizeof(MOPSurfaceProperties);
    m_surfaceBuffer = [m_device newBufferWithBytes:m_surfaces.data()
                                length:size
                                options:MTLResourceStorageModeShared];
}

// C7 수정: SetBorderSurface — volumeId 기반 비교 (materialId가 아닌)
void MetalOpticalEngineImpl::SetBorderSurface(uint32_t vol1, uint32_t vol2, uint32_t surfId) {
    for (auto& tri : m_triangles) {
        if (tri.volumeId == vol1 || tri.volumeId == vol2) {
            tri.surfaceId = surfId;
        }
    }
}

void MetalOpticalEngineImpl::SetSkinSurface(uint32_t vol, uint32_t surfId) {
    for (auto& tri : m_triangles) {
        if (tri.volumeId == vol) {
            tri.surfaceId = surfId;
        }
    }
}

// ============================================================
// Genstep 수집
// ============================================================
void MetalOpticalEngineImpl::AddGenstep(const MOPGenstep& genstep) {
    // Lock-free: 원자적 인덱스로 배열에 직접 삽입
    uint32_t idx = m_genstepCount.fetch_add(1, std::memory_order_relaxed);
    if (idx < GENSTEP_CAPACITY) {
        m_genstepStorage[idx] = genstep;
    } else {
        // Fix: 용량 초과 시 경고 (첫 발생 시만 출력)
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true)) {
            std::cerr << "[MOP] WARNING: Genstep capacity exceeded! "
                      << "GENSTEP_CAPACITY=" << GENSTEP_CAPACITY
                      << ", attempted index=" << idx
                      << ". Excess gensteps will be DROPPED. "
                      << "Consider reducing events per batch or increasing capacity."
                      << std::endl;
        }
    }
}

void MetalOpticalEngineImpl::ResetGensteps() {
    m_genstepCount.store(0, std::memory_order_relaxed);
}

// ============================================================
// Primary opticalphoton 직접 주입 (Beam source path)
// ============================================================
void MetalOpticalEngineImpl::AddPrimaryPhoton(const MOPPhoton& photon) {
    uint32_t idx = m_primaryPhotonCount.fetch_add(1, std::memory_order_relaxed);
    if (idx < PRIMARY_PHOTON_CAPACITY) {
        m_primaryPhotonStorage[idx] = photon;
        // 진단 (2026-04-22): caller가 새 필드를 세팅 안 할 수 있어 명시 초기화
        m_primaryPhotonStorage[idx].reflectedCount = 0;
        m_primaryPhotonStorage[idx].lastHitTriId = -1;  // Chroma 정석 fix init
    } else {
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true)) {
            std::cerr << "[MOP] WARNING: Primary photon capacity exceeded! "
                      << "PRIMARY_PHOTON_CAPACITY=" << PRIMARY_PHOTON_CAPACITY
                      << ", attempted index=" << idx
                      << ". Excess primaries will be DROPPED." << std::endl;
        }
    }
}

void MetalOpticalEngineImpl::ResetPrimaryPhotons() {
    m_primaryPhotonCount.store(0, std::memory_order_relaxed);
}

uint32_t MetalOpticalEngineImpl::GetPrimaryPhotonCount() const {
    return m_primaryPhotonCount.load(std::memory_order_relaxed);
}

uint32_t MetalOpticalEngineImpl::CalculateTotalPhotons() const {
    uint32_t count = m_genstepCount.load(std::memory_order_relaxed);
    uint32_t total = 0;
    for (uint32_t i = 0; i < count && i < GENSTEP_CAPACITY; i++) {
        total += m_genstepStorage[i].numPhotons;
    }
    return total;
}

void MetalOpticalEngineImpl::CalculateGenstepOffsets(std::vector<uint32_t>& offsets) const {
    uint32_t count = m_genstepCount.load(std::memory_order_relaxed);
    if (count > GENSTEP_CAPACITY) count = GENSTEP_CAPACITY;
    offsets.resize(count + 1);
    offsets[0] = 0;
    for (uint32_t i = 0; i < count; i++) {
        offsets[i + 1] = offsets[i] + m_genstepStorage[i].numPhotons;
    }
}

// ============================================================
// GPU 전파 실행 — CPU↔GPU 동기화 최적화
// ============================================================
uint32_t MetalOpticalEngineImpl::Propagate() {
    uint32_t gsCount = m_genstepCount.load(std::memory_order_relaxed);
    uint32_t primaryCount = m_primaryPhotonCount.load(std::memory_order_relaxed);
    if (!m_initialized) return 0;
    // Bail only if BOTH gensteps and primaries are empty (was: gsCount==0)
    if (gsCount == 0 && primaryCount == 0) return 0;
    if (gsCount > GENSTEP_CAPACITY) gsCount = GENSTEP_CAPACITY;
    if (primaryCount > PRIMARY_PHOTON_CAPACITY) primaryCount = PRIMARY_PHOTON_CAPACITY;

    // Fix 78b: propagate 파이프라인 lazy 생성 (function_constant 바인딩)
    EnsurePropagatePipeline();

    @autoreleasepool {
        auto startTime = std::chrono::high_resolution_clock::now();

        // Generation 단계가 실제로 photon buffer에 쓸 양 (gensteps에서 만들어지는 광자 수만)
        uint32_t genstepPhotons = CalculateTotalPhotons();
        // Primary opticalphoton (Beam source)이 추가로 차지할 슬롯
        uint32_t totalPhotons = genstepPhotons + primaryCount;
        if (totalPhotons == 0) return 0;

        // 배치 상한: 두 종류 합쳐서 maxPhotonsPerBatch까지
        if (totalPhotons > m_config.maxPhotonsPerBatch) {
            // primary는 우선 보존, genstep을 잘라서 맞춤
            if (primaryCount >= m_config.maxPhotonsPerBatch) {
                primaryCount = m_config.maxPhotonsPerBatch;
                genstepPhotons = 0;
            } else {
                genstepPhotons = m_config.maxPhotonsPerBatch - primaryCount;
            }
            totalPhotons = genstepPhotons + primaryCount;
        }

        if (m_logLevel >= 3) {
            std::cout << "[MOP] Propagating " << totalPhotons << " photons from "
                      << gsCount << " gensteps" << std::endl;

            // Genstep 진단: 첫 3개 genstep의 위치/물질 출력
            for (uint32_t i = 0; i < std::min(gsCount, (uint32_t)3); i++) {
                const MOPGenstep& gs = m_genstepStorage[i];
                std::cout << "[MOP]   genstep[" << i << "]: pos=(" << gs.posX << "," << gs.posY << "," << gs.posZ
                          << ") matId=" << gs.materialId << " genType=" << gs.genType
                          << " nPhotons=" << gs.numPhotons << std::endl;
            }
            std::cout << "[MOP]   Triangles: " << m_triangles.size()
                      << ", AccelStruct: " << (m_accelerationStructureBuilt ? "built" : "NOT built")
                      << ", WorldSize: " << m_config.worldSizeX << "x" << m_config.worldSizeY << "x" << m_config.worldSizeZ
                      << std::endl;

            // GPU 물질 버퍼의 흡수 길이 진단 (첫 실행만)
            static bool matDumped = false;
            if (!matDumped && m_materialBuffer) {
                matDumped = true;
                // Fix 36: sizeof 검증 — CPU/GPU 구조체 크기 일치 확인
                std::cout << "[MOP]   sizeof(MOPMaterialProperties)=" << sizeof(MOPMaterialProperties)
                          << " sizeof(MOPPropertyTable)=" << sizeof(MOPPropertyTable)
                          << " sizeof(MOPSurfaceProperties)=" << sizeof(MOPSurfaceProperties)
                          << " sizeof(MOPTriangle)=" << sizeof(MOPTriangle)
                          << " sizeof(MOPPhoton)=" << sizeof(MOPPhoton)
                          << std::endl;

                MOPMaterialProperties* matData = (MOPMaterialProperties*)[m_materialBuffer contents];
                // Fix 36: 물질 이름 역매핑 생성
                std::map<uint32_t, std::string> matIdToName;
                for (auto& [name, id] : m_materialNames) {
                    matIdToName[id] = name;
                }

                for (size_t mi = 0; mi < m_materials.size(); mi++) {
                    std::string matName = "UNKNOWN";
                    auto nameIt = matIdToName.find((uint32_t)mi);
                    if (nameIt != matIdToName.end()) matName = nameIt->second;

                    std::cout << "[MOP]   GPU matId=" << mi
                              << " (field=" << matData[mi].materialId << ")"
                              << " name=\"" << matName << "\""
                              << " absLen.count=" << matData[mi].absorptionLength.count
                              << " rindex.count=" << matData[mi].refractiveIndex.count
                              << " isDetector=" << matData[mi].isDetector;
                    if (matData[mi].absorptionLength.count > 0) {
                        std::cout << " absLen[0]=" << matData[mi].absorptionLength.values[0]
                                  << "mm @" << matData[mi].absorptionLength.energies[0] << "eV";
                    }
                    if (matData[mi].refractiveIndex.count > 0) {
                        std::cout << " rindex[0]=" << matData[mi].refractiveIndex.values[0];
                    }
                    // Fix 36: materialId field 불일치 경고
                    if (matData[mi].materialId != (uint32_t)mi) {
                        std::cout << " *** MISMATCH: field=" << matData[mi].materialId
                                  << " vs index=" << mi << " ***";
                    }
                    std::cout << std::endl;
                }
                // Fix 37: 서피스 속성 진단
                if (m_surfaceBuffer) {
                    MOPSurfaceProperties* surfData = (MOPSurfaceProperties*)[m_surfaceBuffer contents];
                    std::map<uint32_t, std::string> surfIdToName;
                    for (auto& [name, id] : m_surfaceNames) { surfIdToName[id] = name; }
                    for (size_t si = 0; si < m_surfaces.size(); si++) {
                        std::string sName = "UNKNOWN";
                        auto nIt = surfIdToName.find((uint32_t)si);
                        if (nIt != surfIdToName.end()) sName = nIt->second;
                        std::cout << "[MOP]   GPU surfId=" << si
                                  << " name=\"" << sName << "\""
                                  << " type=" << surfData[si].type
                                  << " model=" << surfData[si].model
                                  << " finish=" << surfData[si].finish
                                  << " refl.count=" << surfData[si].reflectivity.count
                                  << " trans.count=" << surfData[si].transmittance.count
                                  << " eff.count=" << surfData[si].efficiency.count
                                  << " sigmaAlpha=" << surfData[si].sigmaAlpha
                                  << std::endl;
                    }
                }
                // config 확인
                std::cout << "[MOP]   Config: enableAbsorption=" << m_config.enableAbsorption
                          << " enableBoundary=" << m_config.enableBoundary
                          << " maxSteps=" << m_config.maxStepsPerPhoton << std::endl;
            }
        }

        // === 영속 버퍼 재사용 (필요 시만 재할당) ===
        EnsureGenstepBuffer(gsCount);
        EnsurePhotonBuffer(totalPhotons);
        EnsureHitBuffer(totalPhotons);

        // Fix 69: memset 제거 — SLC 오염 방지
        // 생성 커널이 photon[0..totalPhotons-1]을 전부 덮어쓰므로 memset 불필요
        // propagate 커널은 if (idx >= totalPhotons) return; 으로 범위 보호
        // CPU memset(34MB)가 SLC를 오염시켜 GPU BVH 캐시를 밀어냈음

        // genstep 데이터를 GPU 버퍼에 복사 (Apple Silicon 통합 메모리: memcpy 직접 가능)
        memcpy([m_genstepBuffer contents], m_genstepStorage,
               gsCount * sizeof(MOPGenstep));

        // hit count 리셋
        *((uint32_t*)[m_hitCountBuffer contents]) = 0;
        *((uint32_t*)[m_photonCountBuffer contents]) = 0;

        // Fix 41: Transit hit 버퍼 할당/리셋 + 스코어링 마스크 설정
        {
            // 2026-05-03 fix: 동적 grow capacity (이전 cap 2M 고정 → batch 별 truncation 으로
            // result 가 batch size 에 의존). 처음 50M, 이전 dispatch 의 overflow-detect 시
            // 2x 확장 (m_transitGrowRequested 가 set 되어 있으면).
            // 2026-05-23: floor 1.5B → 104M (4.16GB). 배경: transitCap*40 가 uint32 곱셈
            // 오버플로우로 원래 3.88GB(104M)만 할당하던 버그가 있었고, 그 크기에서 모든
            // 벤치(06 1B Mie 포함)가 정상 동작했다. (size_t) 로 고쳐 의도된 57GB 를 노출하니
            // 06 의 1B 광자 처리 + 대형 버퍼 조합이 SIGKILL(20GB 단독 할당은 OK, maxBufferLength
            // 373GB — 즉 버퍼 크기 자체가 아니라 1B throughput 과의 조합 문제). 검증된 104M 로
            // 고정 (buffer==capacity 일치 → overflow 시 OOB 대신 clean FATAL). 현 벤치 최대 06 ~65M.
            uint32_t transitCap = std::max<uint32_t>(104000000u, m_transitBufferCapacity);
            if (m_transitGrowRequested) {
                transitCap = m_transitBufferCapacity * 2;
                m_transitGrowRequested = false;
            }
            if (m_transitBufferCapacity < transitCap) {
                // 2026-05-12 fix: HazardTrackingModeUntracked 제거 — multi-batch race
                // (batch N+1 propagation 이 batch N DDA dispatch 의 buffer read 와 overlap)
                m_transitHitBuffer = [m_device newBufferWithLength:(size_t)transitCap * 40
                                               options:MTLResourceStorageModeShared];
                m_transitBufferCapacity = transitCap;
                if (m_logLevel >= 1) {
                    std::cout << "[MOP] Transit hit buffer capacity = " << transitCap
                              << " (" << (transitCap * 40ull / (1024*1024)) << " MB)" << std::endl;
                }
            }
            *((uint32_t*)[m_transitCountBuffer contents]) = 0;
            *((uint32_t*)[m_maxTransitHitsBuffer contents]) = m_transitBufferCapacity;

            // SF1 (2026-04-24): Surface hit 버퍼 lazy 할당 + 리셋
            // 활성 surface가 있을 때만 큰 버퍼 할당, 아니면 zero-size로 idle
            uint32_t desiredSurfCap = (m_numActiveSurfaces > 0) ? 2000000u : 1024u;
            if (m_surfaceHitCapacity < desiredSurfCap) {
                m_surfaceHitBuffer = [m_device newBufferWithLength:desiredSurfCap * sizeof(MOPSurfaceHit)
                                                options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked];
                m_surfaceHitCapacity = desiredSurfCap;
            }
            *((uint32_t*)[m_surfaceHitCountBuffer contents]) = 0;
            *((uint32_t*)[m_maxSurfaceHitsBuffer contents]) = m_surfaceHitCapacity;

            // Fix 61: 궤적 카운터 초기화
            *((uint32_t*)[m_trajectoryCountBuffer contents]) = 0;

            // 진단: 경계면 카운터 초기화 (buffer 8192 entries)
            memset([m_diagBuffer contents], 0, 8192 * sizeof(uint32_t));
            // Expert reflect_dir (2026-04-22): slot 85 (dir.x min), slot 87 (dir.z min)
            // atomic_min을 위해 UINT_MAX 초기화. slot 86 (max)은 0 OK.
            {
                uint32_t* dc = (uint32_t*)[m_diagBuffer contents];
                dc[85] = 0xFFFFFFFFu;
                dc[87] = 0xFFFFFFFFu;
            }
        }

        // 이벤트별 시드 변화: 매 Propagate 호출마다 다른 시드
        m_eventCounter++;
        // 2026-05-12: TOPAS Ts/Seed 를 base 로 + per-batch hash. seed 별 다른 결과 보장.
        m_config.randomSeed = m_baseSeed * 0xDEADBEEFu + m_eventCounter * 0x9E3779B9u;

        // Fix: genstep 수를 config에 설정 (생성 커널 OOB 방지)
        m_config.numGensteps = gsCount;

        // 2026-05-12 fix: globalPhotonOffset 누적 — shader InitRandomState(gid + offset, ...)
        // batch 마다 unique RNG seed 보장. 결과가 EventBatchSize 에 의존하지 않도록.
        // (이미 randomSeed 가 batch 마다 변경되지만, 동일 photonId * randomSeed 의 hash
        //  품질이 일정 batch 패턴에서 약하다면 globalPhotonOffset 추가 entropy 가 보강.)
        // 본 propagate 호출 전 offset 값이 그대로 사용됨 (현재 batch 의 시작 인덱스).
        // 호출 끝나면 batch 의 totalPhotons 만큼 누적.

        // config 업데이트
        memcpy([m_configBuffer contents], &m_config, sizeof(MOPSimConfig));

        // Genstep 오프셋 업데이트
        std::vector<uint32_t> offsets;
        CalculateGenstepOffsets(offsets);
        memcpy([m_genstepOffsetBuffer contents], offsets.data(),
               offsets.size() * sizeof(uint32_t));

        // totalPhotons를 별도 버퍼에 기록
        *((uint32_t*)[m_batchCountBuffer contents]) = totalPhotons;

        // === GPU 파이프라인 실행 ===
        {
            id<MTLCommandBuffer> cmdBuf = [m_commandQueue commandBuffer];

            // 1. 신틸레이션 광자 생성 커널 — gensteps 슬롯만 채움 [0..genstepPhotons)
            if (m_generateScintPipeline && genstepPhotons > 0) {
                id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
                [encoder setComputePipelineState:m_generateScintPipeline];
                [encoder setBuffer:m_genstepBuffer        offset:0 atIndex:0];
                [encoder setBuffer:m_materialBuffer       offset:0 atIndex:1];
                [encoder setBuffer:m_photonBuffer         offset:0 atIndex:2];
                [encoder setBuffer:m_configBuffer         offset:0 atIndex:3];
                [encoder setBuffer:m_photonCountBuffer    offset:0 atIndex:4];
                [encoder setBuffer:m_genstepOffsetBuffer  offset:0 atIndex:5];
                [encoder setBuffer:m_photonMetaBuffer     offset:0 atIndex:6];  // Fix 78측정-6
                [encoder setBuffer:m_diagBuffer           offset:0 atIndex:7];  // [EmitDiag]

                NSUInteger threadWidth = [m_generateScintPipeline threadExecutionWidth];
                [encoder dispatchThreads:MTLSizeMake(genstepPhotons, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(threadWidth, 1, 1)];
                [encoder endEncoding];
            }

            // 2. 체렌코프 광자 생성 커널 — gensteps 슬롯만 채움
            if (m_generateCerenkovPipeline && genstepPhotons > 0) {
                id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
                [encoder setComputePipelineState:m_generateCerenkovPipeline];
                [encoder setBuffer:m_genstepBuffer        offset:0 atIndex:0];
                [encoder setBuffer:m_materialBuffer       offset:0 atIndex:1];
                [encoder setBuffer:m_photonBuffer         offset:0 atIndex:2];
                [encoder setBuffer:m_configBuffer         offset:0 atIndex:3];
                [encoder setBuffer:m_photonCountBuffer    offset:0 atIndex:4];
                [encoder setBuffer:m_genstepOffsetBuffer  offset:0 atIndex:5];
                [encoder setBuffer:m_photonMetaBuffer     offset:0 atIndex:6];  // Fix 78측정-6
                [encoder setBuffer:m_diagBuffer           offset:0 atIndex:7];  // [EmitDiag]

                NSUInteger threadWidth = [m_generateCerenkovPipeline threadExecutionWidth];
                [encoder dispatchThreads:MTLSizeMake(genstepPhotons, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(threadWidth, 1, 1)];
                [encoder endEncoding];
            }

            // 2c. 2026-05-12: GPU beam photon generation — Genstep genType=3 인 BEAM 케이스.
            // 단일 BEAM genstep 1개 → numPhotons 광자 생성 (TOPAS Beam source 우회).
            // dispatchThreads = numPhotons. photons[0..numPhotons) 슬롯 채움.
            if (m_generateBeamPipeline && genstepPhotons > 0) {
                id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
                [encoder setComputePipelineState:m_generateBeamPipeline];
                [encoder setBuffer:m_genstepBuffer        offset:0 atIndex:0];
                [encoder setBuffer:m_materialBuffer       offset:0 atIndex:1];
                [encoder setBuffer:m_photonBuffer         offset:0 atIndex:2];
                [encoder setBuffer:m_configBuffer         offset:0 atIndex:3];
                [encoder setBuffer:m_photonCountBuffer    offset:0 atIndex:4];
                [encoder setBuffer:m_genstepOffsetBuffer  offset:0 atIndex:5];
                [encoder setBuffer:m_photonMetaBuffer     offset:0 atIndex:6];
                // 2026-05-17 BEAM kernel material auto-detect — geometry buffer binding.
                // Photon position 기반 cyl/box/sphere inside test → materialId 자동 set.
                [encoder setBuffer:m_cylinderGeometryBuffer offset:0 atIndex:7];
                [encoder setBuffer:m_cylinderGeometryCountBuffer offset:0 atIndex:8];
                [encoder setBuffer:m_boxGeometryBuffer    offset:0 atIndex:9];
                [encoder setBuffer:m_boxGeometryCountBuffer offset:0 atIndex:10];
                [encoder setBuffer:m_sphereGeometryBuffer offset:0 atIndex:11];
                [encoder setBuffer:m_sphereGeometryCountBuffer offset:0 atIndex:12];

                NSUInteger threadWidth = [m_generateBeamPipeline threadExecutionWidth];
                [encoder dispatchThreads:MTLSizeMake(genstepPhotons, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(threadWidth, 1, 1)];
                [encoder endEncoding];
                if (m_logLevel >= 2) {
                    std::cout << "[MOP] BEAM kernel dispatched: " << genstepPhotons << " photons" << std::endl;
                }
            }

            // 2b. Primary opticalphoton 직접 주입 — [genstepPhotons..totalPhotons) 슬롯에 memcpy
            // PhotonGeneration 커널이 끝난 후, 추가 GPU dispatch 없이 host-side 메모리 복사로 처리.
            // Apple Silicon의 unified memory 덕분에 GPU 가시성 즉시 보장 (storage mode shared).
            if (primaryCount > 0 && m_photonBuffer) {
                MOPPhoton* photonBufContents = (MOPPhoton*)[m_photonBuffer contents];
                memcpy(photonBufContents + genstepPhotons,
                       m_primaryPhotonStorage,
                       primaryCount * sizeof(MOPPhoton));
                if (m_logLevel >= 2) {
                    std::cout << "[MOP] Injected " << primaryCount
                              << " primary opticalphoton(s) at slot offset "
                              << genstepPhotons << std::endl;
                }
            }

            // 3. 전파 커널 — dispatchThreads로 정확한 스레드 수 (GPU가 자동 최적 배분)
            // Fix 78측정-5: GPU Frame Capture 카운터 (전파 dispatch 진입 전 선언)
            static int captureTarget = -1;
            static int captureCounter = 0;
            static dispatch_once_t captureOnce;
            dispatch_once(&captureOnce, ^{
                const char* env = getenv("MOP_CAPTURE_DISPATCH");
                if (env) captureTarget = atoi(env);
            });
            bool captureNow = false;
            if (m_propagatePipeline && m_accelStructure) {
                id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
                [encoder setComputePipelineState:m_propagatePipeline];
                [encoder setBuffer:m_photonBuffer         offset:0 atIndex:0];
                [encoder setBuffer:m_materialBuffer       offset:0 atIndex:1];
                [encoder setBuffer:m_surfaceBuffer        offset:0 atIndex:2];
                [encoder setBuffer:m_configBuffer         offset:0 atIndex:3];
                [encoder setBuffer:m_hitBuffer            offset:0 atIndex:4];
                [encoder setBuffer:m_hitCountBuffer       offset:0 atIndex:5];
                [encoder setBuffer:m_triangleBuffer       offset:0 atIndex:6];
                [encoder setBuffer:m_triangleCountBuffer  offset:0 atIndex:7];
                [encoder setBuffer:m_batchCountBuffer     offset:0 atIndex:8];
                [encoder setAccelerationStructure:m_accelStructure atBufferIndex:9];

                // Fix 41: Transit hit 버퍼 바인딩
                [encoder setBuffer:m_transitHitBuffer     offset:0 atIndex:10];
                [encoder setBuffer:m_transitCountBuffer   offset:0 atIndex:11];
                [encoder setBuffer:m_maxTransitHitsBuffer  offset:0 atIndex:12];

                // Fix (2026-04-22): 진단 빌드를 위해 buffer 13-15 binding 활성화.
                // ENABLE_DIAGNOSTICS=1 셰이더에서 trajectory dump + diagCounters 사용.
                [encoder setBuffer:m_diagBuffer            offset:0 atIndex:13];
                [encoder setBuffer:m_trajectoryBuffer      offset:0 atIndex:14];
                [encoder setBuffer:m_trajectoryCountBuffer offset:0 atIndex:15];

                // Fix 64: 재질 LUT 버퍼 바인딩
                [encoder setBuffer:m_materialLUTBuffer      offset:0 atIndex:16];

                // Fix 71: 표면 LUT 버퍼 바인딩
                [encoder setBuffer:m_surfaceLUTBuffer       offset:0 atIndex:17];

                // Fix 78측정-6: cold meta 버퍼 바인딩 (hit 기록 시만 접근)
                [encoder setBuffer:m_photonMetaBuffer       offset:0 atIndex:18];

                // Two-BVH Phase 1: scoring AS 바인딩 (현재 dummy, kernel 미사용)
                if (m_scoringAccelStructure) {
                    [encoder setAccelerationStructure:m_scoringAccelStructure atBufferIndex:20];
                }

                // G1 (Opticks reference): G4Box analytic geometry buffer
                [encoder setBuffer:m_boxGeometryBuffer       offset:0 atIndex:21];
                [encoder setBuffer:m_boxGeometryCountBuffer  offset:0 atIndex:22];

                // SF1 (2026-04-24): GPU surface flux scorer 버퍼 바인딩
                [encoder setBuffer:m_surfaceDefBuffer        offset:0 atIndex:23];
                [encoder setBuffer:m_surfaceHitBuffer        offset:0 atIndex:24];
                [encoder setBuffer:m_surfaceHitCountBuffer   offset:0 atIndex:25];
                [encoder setBuffer:m_maxSurfaceHitsBuffer    offset:0 atIndex:26];

                // Phase 1 (2026-05-05): G4Sphere analytic geometry 버퍼 바인딩
                [encoder setBuffer:m_sphereGeometryBuffer        offset:0 atIndex:27];
                // Phase 4 (2026-05-18): slot 28 = PolyhedraGeometry buffer (Metal slot 31 한도 회피)
                // sphere count 는 cylTorPolyCounts.w 에 packing
                [encoder setBuffer:m_polyhedraGeometryBuffer     offset:0 atIndex:28];

                // Phase 2/3 (2026-05-05): G4Tubs/G4Torus analytic geometry 바인딩
                // buffer 30 = uint4{numCyl, numTor, numPoly, numSphere} 통합 count
                [encoder setBuffer:m_cylinderGeometryBuffer      offset:0 atIndex:29];
                [encoder setBuffer:m_cylinderGeometryCountBuffer offset:0 atIndex:30];
                // Torus geom buffer 는 unused index 19 사용
                [encoder setBuffer:m_torusGeometryBuffer         offset:0 atIndex:19];

                // Fix 78측정-5: GPU Frame Capture 트리거 (Apple Silicon은 ALU/메모리 카운터를
                //   programmatic으로 노출하지 않으므로 Xcode trace 파일로만 분석 가능)
                // export MOP_CAPTURE_DISPATCH=N → N번째 propagate dispatch를 .gputrace로 캡처
                // export MOP_CAPTURE_OUT=/tmp/file.gputrace → 출력 경로
                if (captureTarget >= 0 && captureCounter == captureTarget) {
                    captureNow = true;
                    const char* outEnv = getenv("MOP_CAPTURE_OUT");
                    NSString* outPath = outEnv ?
                        [NSString stringWithUTF8String:outEnv] :
                        @"/tmp/topas_propagate.gputrace";
                    MTLCaptureManager* mgr = [MTLCaptureManager sharedCaptureManager];
                    MTLCaptureDescriptor* desc = [MTLCaptureDescriptor new];
                    desc.captureObject = m_commandQueue;
                    desc.destination = MTLCaptureDestinationGPUTraceDocument;
                    desc.outputURL = [NSURL fileURLWithPath:outPath];
                    NSError* capErr = nil;
                    if ([mgr startCaptureWithDescriptor:desc error:&capErr]) {
                        std::cout << "[MOP] GPU Frame Capture STARTED for dispatch #"
                                  << captureCounter << " → " << [outPath UTF8String] << std::endl;
                    } else {
                        std::cerr << "[MOP] GPU Frame Capture FAILED: "
                                  << (capErr ? [[capErr description] UTF8String] : "unknown")
                                  << "\n  Hint: 환경변수 METAL_CAPTURE_ENABLED=1 필요" << std::endl;
                        captureNow = false;
                    }
                }

                // Fix 78측정-4: threadgroup 크기 sweep용 환경변수 override
                // export MOP_PROP_TG_SIZE=64 등으로 dispatch 단위 조정 가능
                NSUInteger threadWidth = [m_propagatePipeline threadExecutionWidth];
                static NSUInteger envTgSize = 0;
                static dispatch_once_t onceToken;
                dispatch_once(&onceToken, ^{
                    const char* env = getenv("MOP_PROP_TG_SIZE");
                    if (env) {
                        NSUInteger v = (NSUInteger)atoi(env);
                        NSUInteger maxTG = [m_propagatePipeline maxTotalThreadsPerThreadgroup];
                        if (v >= 1 && v <= maxTG) {
                            envTgSize = v;
                            std::cout << "[MOP] propagate threadgroup size override: "
                                      << envTgSize << " (default " << threadWidth
                                      << ", max " << maxTG << ")" << std::endl;
                        }
                    }
                });
                NSUInteger tgSize = envTgSize > 0 ? envTgSize : threadWidth;
                [encoder dispatchThreads:MTLSizeMake(totalPhotons, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
                [encoder endEncoding];
            }

            if (cmdBuf) {
                // Fix 78측정: dispatch별 정확한 GPU 시간 측정 (GPUEndTime - GPUStartTime)
                // commit 전에 handler 등록 — propagate cmdBuf 완료 시 자동 호출
                MetalOpticalEngineImpl* self = this;
                [cmdBuf addCompletedHandler:^(id<MTLCommandBuffer> cb) {
                    double ms = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
                    self->m_stats.gpuPropagateMs += ms;
                    self->m_stats.propagateDispatchCount++;
                }];
                [cmdBuf commit];
                // Fix 78측정-5: GPU Frame Capture 종료
                if (captureNow) {
                    [cmdBuf waitUntilCompleted];  // 캡처 대상 dispatch 완료 보장
                    [[MTLCaptureManager sharedCaptureManager] stopCapture];
                    std::cout << "[MOP] GPU Frame Capture STOPPED. Open .gputrace in Xcode to analyze."
                              << std::endl;
                }
                captureCounter++;
                // Fix 71: 파이프라이닝 — commit 후 즉시 반환, 결과는 WaitForGPU()에서 수확
                m_pendingCmdBuf = cmdBuf;
                m_hasPendingWork = true;
                m_pendingTotalPhotons = totalPhotons;
                m_pendingSubmitTime = std::chrono::high_resolution_clock::now();

                // 2026-05-12 fix: 다음 batch 의 globalPhotonOffset 누적
                // shader InitRandomState(gid + offset, ...) → batch 마다 unique RNG.
                m_config.globalPhotonOffset += totalPhotons;
            }
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        double submitMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
        m_stats.totalTimeMs += submitMs;

        if (m_logLevel >= 3) {
            std::cout << "[MOP] Propagation: submitted "
                      << totalPhotons << " photons, "
                      << submitMs << " ms (async)" << std::endl;

            // 광자 상태 분포 진단 — 0 히트 원인 파악용 (GPU 완료 필요)
            WaitForGPU();
            MOPPhoton* photonData = (MOPPhoton*)[m_photonBuffer contents];
            uint32_t statusCounts[7] = {0}; // ALIVE=0, ABSORBED=1, DETECTED=2, BOUNDARY_ABS=3, OUT_OF_WORLD=4, MAX_STEPS=5, REEMITTED=6
            uint32_t matIdCounts[MOP_MAX_MATERIALS] = {0};
            float samplePosX = 0, samplePosY = 0, samplePosZ = 0;
            uint32_t sampleMatId = 0, sampleSteps = 0, sampleStatus = 0;
            for (uint32_t i = 0; i < totalPhotons; i++) {
                uint32_t s = photonData[i].status;
                if (s < 7) statusCounts[s]++;
                uint32_t m = photonData[i].materialId;
                if (m < MOP_MAX_MATERIALS) matIdCounts[m]++;
                if (i == 0) {
                    samplePosX = photonData[i].posX;
                    samplePosY = photonData[i].posY;
                    samplePosZ = photonData[i].posZ;
                    sampleMatId = photonData[i].materialId;
                    sampleSteps = photonData[i].stepCount;
                    sampleStatus = photonData[i].status;
                }
            }
            const char* statusNames[] = {"ALIVE", "ABSORBED", "DETECTED", "BOUNDARY_ABS", "OUT_OF_WORLD", "MAX_STEPS", "REEMITTED"};
            std::cout << "[MOP] === Photon Status Distribution ===" << std::endl;
            for (int s = 0; s < 7; s++) {
                if (statusCounts[s] > 0) {
                    std::cout << "[MOP]   " << statusNames[s] << ": " << statusCounts[s]
                              << " (" << (100.0 * statusCounts[s] / totalPhotons) << "%)" << std::endl;
                }
            }

            // [FateMatrix 2026-04-27] photon-별 fate 분류 (status × final face/region)
            // Host 가 photon buffer 직접 read (atomic conflict 영향 없음)
            uint32_t f_xm = 0, f_xp = 0, f_ym = 0, f_yp = 0, f_zm = 0, f_zp = 0;
            uint32_t f_abs_bc = 0, f_abs_other = 0;
            // reflectedCount histogram: 0, 1, 2, 3, 4, 5-9, 10-99, 100+
            uint32_t reflBins[8] = {0};
            for (uint32_t i = 0; i < totalPhotons; i++) {
                uint8_t st = photonData[i].status;
                float fx = photonData[i].posX, fy = photonData[i].posY, fz = photonData[i].posZ;
                if (st == 4) {  // OUT_OF_WORLD
                    float ax = std::abs(fx), ay = std::abs(fy), az = std::abs(fz);
                    float dx = ax - 25.0f, dy = ay - 25.0f, dz = az - 150.0f;
                    if (dx >= dy && dx >= dz) {
                        if (fx < 0) f_xm++; else f_xp++;
                    } else if (dy >= dz) {
                        if (fy < 0) f_ym++; else f_yp++;
                    } else {
                        if (fz < 0) f_zm++; else f_zp++;
                    }
                } else if (st == 1) {  // ABSORBED
                    if (photonData[i].materialId == 1u) f_abs_bc++;
                    else f_abs_other++;
                }
                // reflectedCount histogram (photon.flags >> 27 의 5 비트, max 31)
                uint32_t reflCount = (photonData[i].flags >> 27) & 0x1F;
                int rb = 0;
                if (reflCount == 0) rb = 0;
                else if (reflCount == 1) rb = 1;
                else if (reflCount == 2) rb = 2;
                else if (reflCount == 3) rb = 3;
                else if (reflCount == 4) rb = 4;
                else if (reflCount < 10) rb = 5;
                else if (reflCount < 100) rb = 6;
                else rb = 7;
                reflBins[rb]++;
            }
            std::cout << "[FateMatrix-Host] OUT_OF_WORLD face escape:"
                      << " -X=" << f_xm << " +X=" << f_xp
                      << " -Y=" << f_ym << " +Y=" << f_yp
                      << " -Z=" << f_zm << " +Z=" << f_zp << std::endl;
            std::cout << "[FateMatrix-Host]   side(X+Y) sum=" << (f_xm+f_xp+f_ym+f_yp)
                      << "  end(-Z+Z) sum=" << (f_zm+f_zp) << std::endl;
            std::cout << "[FateMatrix-Host] ABSORBED: BC408_inside=" << f_abs_bc
                      << " other=" << f_abs_other << std::endl;
            // [ReflPerZ 2026-04-27] reflectedCount per z-bin (forward downstream mechanism)
            uint32_t refl_z[10][8] = {0};  // 10 z-bin × 8 reflBin
            for (uint32_t i = 0; i < totalPhotons; i++) {
                float fz = photonData[i].posZ;
                int zb = (int)((fz + 200.0f) / 60.0f);  // World z [-200, +400] → 10 bin × 60mm
                if (zb < 0) zb = 0;
                if (zb >= 10) zb = 9;
                uint32_t reflCount = (photonData[i].flags >> 27) & 0x1F;
                int rb = 0;
                if (reflCount == 0) rb = 0;
                else if (reflCount == 1) rb = 1;
                else if (reflCount == 2) rb = 2;
                else if (reflCount == 3) rb = 3;
                else if (reflCount == 4) rb = 4;
                else if (reflCount < 10) rb = 5;
                else if (reflCount < 100) rb = 6;
                else rb = 7;
                refl_z[zb][rb]++;
            }
            std::cout << "[ReflPerZ] reflectedCount histogram per final-z bin (60mm bins, World z):" << std::endl;
            std::cout << "[ReflPerZ]   z_bin   r=0    r=1    r=2    r=3    r=4   r=5-9  r=10-99" << std::endl;
            for (int zb = 0; zb < 10; zb++) {
                int z_lo = -200 + zb * 60;
                std::cout << "[ReflPerZ]   z=" << z_lo << ":"
                          << std::setw(8) << refl_z[zb][0]
                          << std::setw(8) << refl_z[zb][1]
                          << std::setw(8) << refl_z[zb][2]
                          << std::setw(8) << refl_z[zb][3]
                          << std::setw(8) << refl_z[zb][4]
                          << std::setw(8) << refl_z[zb][5]
                          << std::setw(8) << refl_z[zb][6] << std::endl;
            }
            std::cout << "[FateMatrix-Host] reflectedCount histogram (per photon, max 31):"
                      << " r=0:" << reflBins[0]
                      << " r=1:" << reflBins[1]
                      << " r=2:" << reflBins[2]
                      << " r=3:" << reflBins[3]
                      << " r=4:" << reflBins[4]
                      << " r=5-9:" << reflBins[5]
                      << " r=10-99:" << reflBins[6]
                      << " r>=100:" << reflBins[7] << std::endl;

            // [RNGDiag 2026-04-27] emit direction (dir.z) chi-square test
            // 광자 emit 후 propagation 가기 전 dir.z 의 isotropic 분포 검증
            // (이미 propagation 후 buffer 라 final dir 이지만, multi-bounce 가 cosθ
            // 분포 보존 — 가정: 첫 N 광자의 dir.z 분포가 ~uniform 이면 RNG OK)
            // 단순화: 모든 photon 의 final dir.z 분포 (uniform if random isotropic)
            uint32_t cosTheta_bins[10] = {0};
            for (uint32_t i = 0; i < totalPhotons; i++) {
                float dz = photonData[i].dirZ;
                if (dz < -1.0f) dz = -1.0f;
                if (dz >  1.0f) dz =  1.0f;
                int bin = (int)((dz + 1.0f) * 5.0f);  // [-1, 1] → [0, 9]
                if (bin < 0) bin = 0;
                if (bin > 9) bin = 9;
                cosTheta_bins[bin]++;
            }
            // chi-square against uniform expectation (totalPhotons / 10 per bin)
            double expected = (double)totalPhotons / 10.0;
            double chi2 = 0.0;
            for (int i = 0; i < 10; i++) {
                double diff = cosTheta_bins[i] - expected;
                chi2 += (diff * diff) / expected;
            }
            std::cout << "[RNGDiag] dir.z histogram (10 bins, cosθ ∈ [-1, +1]):" << std::endl;
            for (int i = 0; i < 10; i++) {
                double pct = 100.0 * cosTheta_bins[i] / totalPhotons;
                std::cout << "[RNGDiag]   bin " << i << " ["
                          << (-1.0 + i * 0.2) << "," << (-1.0 + (i+1) * 0.2)
                          << "]  count=" << cosTheta_bins[i]
                          << "  (" << pct << "%, expected 10%)" << std::endl;
            }
            std::cout << "[RNGDiag] chi-square=" << chi2
                      << "  (uniform if << 16.92 = 9 dof 95% threshold)" << std::endl;

            // [EmitDiag-Host 2026-04-27] emit z 분포 sub-bin (50 bin × 6mm 으로 aggregation)
            uint32_t* dc_emit = (uint32_t*)[m_diagBuffer contents];
            uint32_t emit_zbin[50] = {0};
            uint64_t emit_total = 0;
            for (int b = 0; b < 300; b++) {
                int zb = b / 6;  // 1mm bin → 6mm bin
                if (zb >= 50) zb = 49;
                emit_zbin[zb] += dc_emit[1300 + b];
                emit_total += dc_emit[1300 + b];
            }
            std::cout << "[EmitDiag] emit z distribution (50 bins × 6mm, total=" << emit_total << "):" << std::endl;
            for (int b = 0; b < 50; b += 4) {
                double zc = -150 + (b + 0.5) * 6.0;
                std::cout << "[EmitDiag]   z=" << zc << "mm  count=" << emit_zbin[b] << std::endl;
            }

            // [EmitDirDiag-GPU 2026-04-28] cosθ + φ histogram (slot 5000-5037)
            uint64_t cosz_total = 0, phi_total = 0;
            for (int b = 0; b < 20; b++) cosz_total += dc_emit[5000 + b];
            for (int b = 0; b < 18; b++) phi_total += dc_emit[5020 + b];
            std::cout << "[EmitDirDiag-GPU] cosθ histogram (20 bins ∈ [-1,+1], total="
                      << cosz_total << ", isotropic = 5.000%/bin):" << std::endl;
            for (int b = 0; b < 20; b++) {
                double lo = -1.0 + b * 0.1;
                double pct = (cosz_total > 0) ? 100.0 * dc_emit[5000 + b] / cosz_total : 0;
                std::cout << "[EmitDirDiag-GPU]   cosθ=[" << lo << "," << lo+0.1 << "]: "
                          << dc_emit[5000 + b] << " (" << pct << "%)" << std::endl;
            }
            std::cout << "[EmitDirDiag-GPU] φ histogram (18 bins ∈ [-π,+π], total="
                      << phi_total << ", isotropic = 5.5556%/bin):" << std::endl;
            for (int b = 0; b < 18; b++) {
                double pct = (phi_total > 0) ? 100.0 * dc_emit[5020 + b] / phi_total : 0;
                std::cout << "[EmitDirDiag-GPU]   φ_bin" << b << ": "
                          << dc_emit[5020 + b] << " (" << pct << "%)" << std::endl;
            }
            // Fix 78e: _padding 인코딩 제거 — 샘플 광자 기본 정보만 출력
            for (uint32_t si = 0; si < std::min(totalPhotons, (uint32_t)5); si++) {
                uint32_t volBits;
                memcpy(&volBits, &photonData[si].weight, sizeof(uint32_t));
                std::cout << "[MOP]   photon[" << si << "]: pos=(" << photonData[si].posX << "," << photonData[si].posY << "," << photonData[si].posZ
                          << ") matId=" << photonData[si].materialId << " steps=" << photonData[si].stepCount
                          << " flags=" << photonData[si].flags
                          << " E=" << photonData[si].energy << "eV"
                          << " volBits=0x" << std::hex << volBits << std::dec
                          << " status=" << (photonData[si].status < 7 ? statusNames[photonData[si].status] : "UNKNOWN") << std::endl;
            }
            // Fix 78측정-6: 옛 검증 코드 제거 (genstepId/parentTrackId가 PhotonMeta 로 이동, 2026-05-12 cleanup)
            // Fix 78e: _padding 인코딩 제거 — hitTri 역추적 기능은 필요 시 별도 버퍼로 재도입
            // Fix 36: 볼륨별 경계 충돌 통계
            uint32_t photonsHitVol[16] = {0};
            for (uint32_t i = 0; i < totalPhotons; i++) {
                uint32_t volBits;
                memcpy(&volBits, &photonData[i].weight, sizeof(uint32_t));
                for (int v = 0; v < 16; v++) {
                    if (volBits & (1u << v)) photonsHitVol[v]++;
                }
            }
            std::cout << "[MOP]   Volume boundary hits:";
            for (int v = 0; v < 16; v++) {
                if (photonsHitVol[v] > 0)
                    std::cout << " vol" << v << "=" << photonsHitVol[v];
            }
            std::cout << std::endl;
            // MAX_STEPS 광자의 평균 flags(bounces) 계산
            uint64_t totalBounces = 0;
            uint32_t maxStepCount = 0;
            for (uint32_t i = 0; i < totalPhotons; i++) {
                if (photonData[i].status == 5) { // MAX_STEPS
                    totalBounces += photonData[i].flags;
                    maxStepCount++;
                }
            }
            if (maxStepCount > 0) {
                std::cout << "[MOP]   MAX_STEPS photons avg bounces: " << (totalBounces / maxStepCount) << std::endl;
            }
            // 물질별 분포 (비영 항목만)
            for (uint32_t m = 0; m < MOP_MAX_MATERIALS; m++) {
                if (matIdCounts[m] > 0) {
                    std::cout << "[MOP]   matId=" << m << ": " << matIdCounts[m] << " photons" << std::endl;
                }
            }
            // Fix 31 진단: Z-range 분포 (광자가 어디까지 도달하는지 확인)
            // Z 범위별 최종 위치 분포 (10개 구간)
            uint32_t zBins[12] = {0}; // [0]:<-300, [1]:-300~-250, [2]:-250~-200, ..., [10]:>100, [11]:nan/inf
            float minZ = 1e10, maxZ = -1e10;
            uint32_t minZReachedCount = 0; // Z < -250에 도달한 광자 수
            for (uint32_t i = 0; i < totalPhotons; i++) {
                float z = photonData[i].posZ;
                if (std::isnan(z) || std::isinf(z)) { zBins[11]++; continue; }
                if (z < minZ) minZ = z;
                if (z > maxZ) maxZ = z;
                if (z < -250.0f) minZReachedCount++;
                int bin = (int)((z + 300.0f) / 50.0f);
                if (bin < 0) bin = 0;
                if (bin > 10) bin = 10;
                zBins[bin]++;
            }
            std::cout << "[MOP]   Z-range: min=" << minZ << " max=" << maxZ
                      << " photons_below_Z-250=" << minZReachedCount << std::endl;
            std::cout << "[MOP]   Z-distribution:";
            for (int b = 0; b < 12; b++) {
                if (zBins[b] > 0) {
                    float zLow = -300.0f + b * 50.0f;
                    if (b == 0) std::cout << " [<-300]:" << zBins[b];
                    else if (b == 11) std::cout << " [nan]:" << zBins[b];
                    else std::cout << " [" << (int)zLow << "," << (int)(zLow+50) << "):" << zBins[b];
                }
            }
            std::cout << std::endl;
            // Z<-280 도달 광자의 상세 정보 덤프 (렌즈/센서 영역 근처)
            uint32_t deepTotal = 0;
            uint32_t deepDumpCount = 0;
            uint32_t deepInsideLensAperture = 0; // r < 10mm인 deep 광자 수
            uint32_t deepInsideHole = 0; // r < 15mm인 deep 광자 수
            uint32_t volIdCounts[16] = {0};
            for (uint32_t i = 0; i < totalPhotons; i++) {
                float z = photonData[i].posZ;
                if (z < -280.0f) {
                    deepTotal++;
                    float r = sqrt(photonData[i].posX * photonData[i].posX +
                                   photonData[i].posY * photonData[i].posY);
                    if (r < 15.0f) deepInsideHole++;
                    if (r < 10.0f) deepInsideLensAperture++;
                    uint32_t vid = photonData[i].volumeId;
                    if (vid < 16) volIdCounts[vid]++;
                    // r<15mm인 광자는 모두 출력 (hole 통과 후보)
                    if (r < 15.0f || deepDumpCount < 10) {
                        std::cout << "[MOP]   DEEP photon[" << i << "]: pos=("
                                  << photonData[i].posX << "," << photonData[i].posY << "," << photonData[i].posZ
                                  << ") r=" << r
                                  << " matId=" << photonData[i].materialId
                                  << " volId=" << photonData[i].volumeId
                                  << " steps=" << photonData[i].stepCount
                                  << " flags=" << photonData[i].flags
                                  << " dir=(" << photonData[i].dirX << "," << photonData[i].dirY << "," << photonData[i].dirZ << ")"
                                  << " status=" << statusNames[photonData[i].status] << std::endl;
                        deepDumpCount++;
                    }
                }
            }
            std::cout << "[MOP]   DEEP: total=" << deepTotal
                      << " r<15mm(hole)=" << deepInsideHole
                      << " r<10mm(lens)=" << deepInsideLensAperture << std::endl;
            std::cout << "[MOP]   DEEP volId:";
            for (int v = 0; v < 16; v++) {
                if (volIdCounts[v] > 0)
                    std::cout << " vol" << v << "=" << volIdCounts[v];
            }
            std::cout << std::endl;

        }

        // 2026-05-03: 진단 블록 전체 logLevel 가드 (이전 "항상 출력" 주석 무시 — production 에서 60만+ 줄 출력으로 wall time 5× 증가).
        // 2026-05-15: >= 1 → >= 2. default LogLevel 1 에서 매 dispatch 진단 출력
        //   (100k event = 10MB 로그, dispatch 1700회 반복) → disk I/O 로 GPU wall time 3× 증가.
        //   per-dispatch 진단은 LogLevel 2+ 에서만 출력.
        if (m_logLevel >= 2) {
        // 진단: 경계면 이벤트 카운터 출력
        {
            uint32_t* dc = (uint32_t*)[m_diagBuffer contents];
            uint32_t surfTotal = dc[0], surfReflect = dc[1], surfTransmit = dc[2];
            uint32_t noSurfTotal = dc[3], noSurfReflect = dc[4], noSurfTransmit = dc[5];
            std::cout << "[MOP] === Boundary Diagnostics ===" << std::endl;
            std::cout << "[MOP]   WithSurface: total=" << surfTotal
                      << " reflect=" << surfReflect
                      << " transmit=" << surfTransmit;
            if (surfTotal > 0)
                std::cout << " (T%=" << (100.0 * surfTransmit / surfTotal) << "%)";
            std::cout << std::endl;
            std::cout << "[MOP]   NoSurface:   total=" << noSurfTotal
                      << " reflect=" << noSurfReflect
                      << " transmit=" << noSurfTransmit;
            if (noSurfTotal > 0)
                std::cout << " (T%=" << (100.0 * noSurfTransmit / noSurfTotal) << "%)";
            std::cout << std::endl;
            uint32_t sameTriReflect = dc[6], nearHit = dc[7];
            uint32_t totalSteps = dc[8];
            std::cout << "[MOP]   SameTriReflect=" << sameTriReflect
                      << " NearHit(<0.001mm)=" << nearHit
                      << " TotalSteps=" << totalSteps << std::endl;

            // CPU TOPAS 비교용 정확한 reflect 카운터 (real branches inside ProcessBoundary*)
            // CPU 기준: FresnelRefraction, FresnelReflection, TIR, SameMaterial
            uint32_t fresTrueRefl = dc[32];
            uint32_t fresTrueRefr = dc[33];
            uint32_t fresTrueTIR  = dc[34];
            uint32_t sameMatExit  = dc[35];
            uint32_t surfTrueRefl = dc[36];
            uint32_t surfTrueRefr = dc[37];
            uint32_t surfTrueTIR  = dc[38];
            uint32_t surfTransOnly= dc[39];
            uint64_t totalReflect = (uint64_t)fresTrueRefl + (uint64_t)fresTrueTIR
                                  + (uint64_t)surfTrueRefl + (uint64_t)surfTrueTIR;
            std::cout << "[MOP] === CPU-comparable Reflect Counters ===" << std::endl;
            std::cout << "[MOP]   FresnelReflection(true)=" << fresTrueRefl
                      << " FresnelRefraction(true)=" << fresTrueRefr
                      << " FresnelTIR(true)=" << fresTrueTIR
                      << " SameMaterial(early-exit)=" << sameMatExit << std::endl;
            std::cout << "[MOP]   WithSurfaceReflect=" << surfTrueRefl
                      << " WithSurfaceRefract=" << surfTrueRefr
                      << " WithSurfaceTIR=" << surfTrueTIR
                      << " WithSurfaceStraightTransmit=" << surfTransOnly << std::endl;
            std::cout << "[MOP]   TOTAL_REFLECT_EVENTS=" << totalReflect
                      << " (FresRefl+FresTIR+SurfRefl+SurfTIR)" << std::endl;

            // 2026-05-09: Lens (dielectric_dielectric) Fresnel branch counters
            uint32_t lensFresRefl = dc[170];
            uint32_t lensTIR      = dc[171];
            uint32_t lensTransmit = dc[172];
            uint64_t lensTotal = (uint64_t)lensFresRefl + (uint64_t)lensTIR + (uint64_t)lensTransmit;
            std::cout << "[MOP] === Lens Fresnel branches (dielectric_dielectric) ==="
                      << std::endl;
            std::cout << "[MOP]   FresnelReflect=" << lensFresRefl
                      << " TIR=" << lensTIR
                      << " Transmit=" << lensTransmit
                      << " Total=" << lensTotal << std::endl;
            if (lensTotal > 0) {
                std::cout << "[MOP]   Fractions: Refl="
                          << (100.0*lensFresRefl/lensTotal) << "% TIR="
                          << (100.0*lensTIR/lensTotal) << "% Transmit="
                          << (100.0*lensTransmit/lensTotal) << "%" << std::endl;
            }
            // ProcessBoundaryWithSurface entry counter
            uint32_t wsEntries = dc[180];
            uint32_t wsHasRefl = dc[181];
            std::cout << "[MOP] WithSurface entries=" << wsEntries
                      << " (with reflectivity=" << wsHasRefl
                      << ", no reflectivity=" << (wsEntries - wsHasRefl) << ")" << std::endl;
            // Lens WithSurface (no-reflectivity) Fresnel branches
            uint32_t wsFresRefl = dc[173];
            uint32_t wsTIR      = dc[174];
            uint32_t wsTransmit = dc[175];
            uint64_t wsTotal = (uint64_t)wsFresRefl + (uint64_t)wsTIR + (uint64_t)wsTransmit;
            std::cout << "[MOP] === Lens WithSurface Fresnel branches ==="
                      << std::endl;
            std::cout << "[MOP]   FresnelReflect=" << wsFresRefl
                      << " TIR=" << wsTIR
                      << " Transmit=" << wsTransmit
                      << " Total=" << wsTotal << std::endl;
            if (wsTotal > 0) {
                std::cout << "[MOP]   Fractions: Refl="
                          << (100.0*wsFresRefl/wsTotal) << "% TIR="
                          << (100.0*wsTIR/wsTotal) << "% Transmit="
                          << (100.0*wsTransmit/wsTotal) << "%" << std::endl;
            }

            // Expert reflect_dir (2026-04-22): Glass bottom (-Z face) reflect dir 정밀도 측정
            // 이론값 (PolSet+Y0, refract dir=(0.4714,0,-0.8820), bottom z=-100 reflect):
            //   newDir = (0.4714, 0, +0.8820)
            // slot 80~87: x_sum, z_sum, x²_sum, z²_sum, count, x_min, x_max, z_min
            {
                uint64_t x_sum_scaled = dc[80];
                uint64_t z_sum_scaled = dc[81];
                uint64_t xx_sum_scaled = dc[82];
                uint64_t zz_sum_scaled = dc[83];
                uint32_t cnt = dc[84];
                uint32_t x_min_scaled = dc[85];
                uint32_t x_max_scaled = dc[86];
                uint32_t z_min_scaled = dc[87];
                std::cout << "[MOP] === Expert reflect_dir (Glass-bottom Fresnel reflect dir) ===" << std::endl;
                std::cout << "[MOP]   GlassBottomReflectCount=" << cnt;
                if (cnt > 0) {
                    double xSumF = (double)x_sum_scaled - (double)cnt * 1.0e5;  // unscale offset
                    double zSumF = (double)z_sum_scaled - (double)cnt * 1.0e5;
                    double meanX = xSumF / 1.0e5 / (double)cnt;
                    double meanZ = zSumF / 1.0e5 / (double)cnt;
                    double meanXX = (double)xx_sum_scaled / 1.0e6 / (double)cnt;
                    double meanZZ = (double)zz_sum_scaled / 1.0e6 / (double)cnt;
                    double varX = meanXX - meanX * meanX;
                    double varZ = meanZZ - meanZ * meanZ;
                    double sdX = (varX > 0) ? std::sqrt(varX) : 0.0;
                    double sdZ = (varZ > 0) ? std::sqrt(varZ) : 0.0;
                    double xMin = ((double)x_min_scaled - 1.0e5) / 1.0e5;
                    double xMax = ((double)x_max_scaled - 1.0e5) / 1.0e5;
                    double zMin = ((double)z_min_scaled - 1.0e5) / 1.0e5;
                    std::cout << std::scientific;
                    std::cout << "\n[MOP]   dir.x mean = " << meanX << " (theoretical 0.4714)"
                              << "  delta = " << (meanX - 0.4714) << std::endl;
                    std::cout << "[MOP]   dir.z mean = " << meanZ << " (theoretical 0.8820)"
                              << "  delta = " << (meanZ - 0.8820) << std::endl;
                    std::cout << "[MOP]   dir.x stddev = " << sdX
                              << "   dir.z stddev = " << sdZ << std::endl;
                    std::cout << "[MOP]   dir.x [min,max] = [" << xMin << ", " << xMax << "]"
                              << "   dir.z min = " << zMin << std::endl;
                    std::cout.unsetf(std::ios::scientific);
                }
                std::cout << std::endl;
            }

            // 진단 (2026-04-22): photon.reflectedCount 합산 — per-photon 분포
            // GPU buffer 직접 reduce (logLevel 무관, primary 200K 비교 시 핵심 지표)
            WaitForGPU();
            MOPPhoton* photonDataR = (MOPPhoton*)[m_photonBuffer contents];
            uint64_t sumRefl = 0;
            uint32_t maxRefl = 0;
            uint32_t hist[8] = {0}; // 0,1,2,3,4,5-9,10-99,>=100
            for (uint32_t i = 0; i < totalPhotons; i++) {
                uint32_t r = photonDataR[i].reflectedCount;
                sumRefl += r;
                if (r > maxRefl) maxRefl = r;
                if (r == 0) hist[0]++;
                else if (r == 1) hist[1]++;
                else if (r == 2) hist[2]++;
                else if (r == 3) hist[3]++;
                else if (r == 4) hist[4]++;
                else if (r < 10) hist[5]++;
                else if (r < 100) hist[6]++;
                else hist[7]++;
            }
            std::cout << "[MOP] === Per-Photon reflectedCount Reduce ===" << std::endl;
            std::cout << "[MOP]   totalPhotons=" << totalPhotons
                      << " sumReflectedCount=" << sumRefl
                      << " maxReflectedCount=" << maxRefl
                      << " avgReflPerPhoton=" << (totalPhotons > 0 ? (double)sumRefl / totalPhotons : 0.0)
                      << std::endl;
            std::cout << "[MOP]   reflHist: r=0:" << hist[0]
                      << " r=1:" << hist[1] << " r=2:" << hist[2]
                      << " r=3:" << hist[3] << " r=4:" << hist[4]
                      << " r=5-9:" << hist[5] << " r=10-99:" << hist[6]
                      << " r>=100:" << hist[7] << std::endl;
            const char* snames[] = {"ALIVE","ABSORBED","DETECTED","BOUNDARY_ABS","OUT_OF_WORLD","MAX_STEPS","REEMITTED"};
            std::cout << "[MOP]   FinalStatus:";
            for (int s = 0; s < 7; s++) {
                if (dc[9+s] > 0) std::cout << " " << snames[s] << "=" << dc[9+s];
            }
            std::cout << std::endl;

            // [FateDiag 2026-04-27] photon fate 분류: face escape + bulk absorbed
            uint32_t f_xm = dc[130], f_xp = dc[131], f_ym = dc[132], f_yp = dc[133];
            uint32_t f_zm = dc[134], f_zp = dc[135];
            uint32_t f_abs_bc = dc[136], f_abs_other = dc[137];
            std::cout << "[FateDiag] OUT_OF_WORLD face escape: -X=" << f_xm
                      << " +X=" << f_xp << " -Y=" << f_ym << " +Y=" << f_yp
                      << " -Z=" << f_zm << " +Z=" << f_zp << std::endl;
            std::cout << "[FateDiag]   side(X+Y)=" << (f_xm+f_xp+f_ym+f_yp)
                      << "  end(-Z+Z)=" << (f_zm+f_zp) << std::endl;
            std::cout << "[FateDiag] ABSORBED: BC408_inside=" << f_abs_bc
                      << " other=" << f_abs_other << std::endl;

            // [Fix82-DEBUG 2026-04-25] Bulk absorb breakdown by matId
            // SF1 R_sf1 fires when photon crosses ScRefl/ZPlusSurface going_in.
            // If absorbInScRefl > R_sf1 fires, missing crossings = bug.
            uint32_t absMat2 = dc[110];  // BC408Black (ScRefl)
            uint32_t absMat1 = dc[111];  // Buapfcfm (Sci/Wrap)
            uint32_t absOther = dc[112];
            std::cout << "[MOP]   BulkAbsorbByMatId: matId=2(BC408Black/ScRefl)=" << absMat2
                      << " matId=1(Buapfcfm/Sci+Wrap)=" << absMat1
                      << " other=" << absOther << std::endl;
            uint32_t matChange12 = dc[120];
            uint32_t mc_stepZero = dc[121];
            uint32_t mc_preInScRefl = dc[122];
            uint32_t mc_postNotInScRefl = dc[123];
            uint32_t mc_noSignChange = dc[124];
            std::cout << "[MOP]   MatChange Sci(1)->ScRefl(2): total=" << matChange12
                      << "  stepDist=0:" << mc_stepZero
                      << "  preInScRefl(z<-100):" << mc_preInScRefl
                      << "  postNotInScRefl(z>=-100):" << mc_postNotInScRefl
                      << "  noSignChange:" << mc_noSignChange << std::endl;

            // Fix 56: 법선 방향 진단
            uint32_t sameMatTotal = dc[21], sameMatSurf = dc[22];
            std::cout << "[MOP] === Normal Direction Diagnostics ===" << std::endl;
            std::cout << "[MOP]   matIdIn==matIdOut: total=" << sameMatTotal
                      << " withSurface=" << sameMatSurf << std::endl;
            std::cout << "[MOP]   WithSurface by volumeId:";
            for (int v = 0; v < 4; v++) {
                if (dc[23+v] > 0) std::cout << " vol" << v << "=" << dc[23+v];
            }
            std::cout << std::endl;

            // 2026-05-07 진단: SDF (torus) hit detection 통계 (slot 150-155)
            {
                uint32_t bs_rej = dc[150], bs_pass = dc[151];
                uint32_t tb_no  = dc[152], it_no  = dc[153];
                uint32_t sdf_hit = dc[154], sum_iter = dc[155];
                uint32_t total = bs_pass;
                std::cout << "[MOP] === Torus SDF Hit Detection (Phase 3.5) ===" << std::endl;
                std::cout << "[MOP]   bsphere_reject=" << bs_rej
                          << "  bsphere_pass=" << bs_pass << std::endl;
                if (total > 0) {
                    std::cout << "[MOP]   sdf_hit_found=" << sdf_hit
                              << " (" << (100.0*sdf_hit/total) << "%)"
                              << "  tbound_break_no_hit=" << tb_no
                              << " (" << (100.0*tb_no/total) << "%)"
                              << "  iter_limit_no_hit=" << it_no
                              << " (" << (100.0*it_no/total) << "%)" << std::endl;
                }
                if (sdf_hit > 0) {
                    std::cout << "[MOP]   avg_iter_to_hit=" << ((double)sum_iter / sdf_hit) << std::endl;
                }
            }

            // 2026-05-06 진단: dielectric_metal reflect dir/normal 평균 (slot 140-148)
            {
                uint32_t sDx = dc[140], sDy = dc[141], sDz = dc[142];
                uint32_t sNx = dc[143], sNy = dc[144], sNz = dc[145];
                uint32_t cnt = dc[146];
                uint32_t zPos = dc[147], zNeg = dc[148];
                if (cnt > 0) {
                    const double sc = 1000.0;
                    double meanDx = (double)sDx / sc / cnt - 1.0;
                    double meanDy = (double)sDy / sc / cnt - 1.0;
                    double meanDz = (double)sDz / sc / cnt - 1.0;
                    double meanNx = (double)sNx / sc / cnt - 1.0;
                    double meanNy = (double)sNy / sc / cnt - 1.0;
                    double meanNz = (double)sNz / sc / cnt - 1.0;
                    std::cout << "[MOP] === Metal Reflect Stats (dielectric_metal) ===" << std::endl;
                    std::cout << "[MOP]   count=" << cnt
                              << " meanNormal=(" << meanNx << "," << meanNy << "," << meanNz << ")"
                              << " meanReflDir=(" << meanDx << "," << meanDy << "," << meanDz << ")"
                              << std::endl;
                    std::cout << "[MOP]   reflDir.z>0: " << zPos << " (" << (100.0*zPos/cnt) << "%)"
                              << "  reflDir.z<=0: " << zNeg << " (" << (100.0*zNeg/cnt) << "%)" << std::endl;
                }
            }

            // HW RT intersection precision diagnostics
            uint32_t bndHitCount = dc[16];
            uint32_t maxDistUint = dc[17];
            uint32_t sumDistNm = dc[18];
            uint32_t bigDistCount = dc[19];  // |distToPlane| > 1μm
            uint32_t veryBigDistCount = dc[20];  // |distToPlane| > 10μm
            float maxDistMM = 0;
            memcpy(&maxDistMM, &maxDistUint, sizeof(float));  // as_type reinterpret
            std::cout << "[MOP] === HW RT Intersection Precision ===" << std::endl;
            std::cout << "[MOP]   BoundaryHits=" << bndHitCount
                      << " maxDistToPlane=" << (maxDistMM * 1e3) << " μm"
                      << " avgDistToPlane=" << (bndHitCount > 0 ? (double)sumDistNm / bndHitCount : 0) << " nm" << std::endl;
            std::cout << "[MOP]   >1μm=" << bigDistCount
                      << " >10μm=" << veryBigDistCount << std::endl;

            // Fix 60: AABB 스코어링 검증 진단
            // 주의: ENABLE_DIAGNOSTICS=0이면 GPU가 dc[28-31]에 기록하지 않음
            uint32_t aabbPathNm = dc[28];      // AABB 교차 경로 합 (nm)
            uint32_t aabbStepCount = dc[29];    // AABB 교차 스텝 수
            uint32_t npPathNm = dc[30];         // newPos 내부 스텝 경로 합 (nm)
            uint32_t npStepCount = dc[31];      // newPos 내부 스텝 수
            double aabbPathMM = (double)aabbPathNm * 1e-6;
            double npPathMM = (double)npPathNm * 1e-6;
            std::cout << "[MOP] === Fix 60: AABB Scoring Verification ===" << std::endl;
            std::cout << "[MOP]   AABBIntersectPath=" << aabbPathMM << " mm"
                      << " steps=" << aabbStepCount
                      << " | NewPosInsidePath=" << npPathMM << " mm"
                      << " steps=" << npStepCount << std::endl;

            // R4 v5 진단: BVH grazing/corner reject 카운터
            std::cout << "[MOP] === R4 BVH retry diag === "
                      << "grazing_reject=" << dc[70]
                      << " corner_reject=" << dc[71]
                      << " accepted=" << dc[72] << std::endl;


            // Expert E (2026-04-22): Glass exit face 분포
            // slots 50-53: face별 attempts, 54-57: face별 reflected, 58/59: +X/-X 분리
            uint32_t gxBottom    = dc[50];
            uint32_t gxTop       = dc[51];
            uint32_t gxSideX     = dc[52];
            uint32_t gxSideY     = dc[53];
            uint32_t gxBottomR   = dc[54];
            uint32_t gxTopR      = dc[55];
            uint32_t gxSideXR    = dc[56];
            uint32_t gxSideYR    = dc[57];
            uint32_t gxPlusX     = dc[58];
            uint32_t gxMinusX    = dc[59];
            uint64_t gxTotal = (uint64_t)gxBottom + gxTop + gxSideX + gxSideY;
            uint64_t gxTotalR = (uint64_t)gxBottomR + gxTopR + gxSideXR + gxSideYR;
            std::cout << "[MOP] === Expert E: Glass Exit Face Distribution ===" << std::endl;
            std::cout << "[MOP]   GlassExitAttempts: total=" << gxTotal
                      << " bottom(-Z)=" << gxBottom
                      << " top(+Z)=" << gxTop
                      << " sideX(±X)=" << gxSideX
                      << " sideY(±Y)=" << gxSideY << std::endl;
            if (gxTotal > 0) {
                std::cout << "[MOP]   GlassExitFracPct: bottom="
                          << (100.0 * gxBottom / gxTotal) << "%"
                          << " top=" << (100.0 * gxTop / gxTotal) << "%"
                          << " sideX=" << (100.0 * gxSideX / gxTotal) << "%"
                          << " sideY=" << (100.0 * gxSideY / gxTotal) << "%" << std::endl;
            }
            std::cout << "[MOP]   GlassExitReflected: total=" << gxTotalR
                      << " bottom=" << gxBottomR
                      << " top=" << gxTopR
                      << " sideX=" << gxSideXR
                      << " sideY=" << gxSideYR << std::endl;
            std::cout << "[MOP]   GlassExitReflFracPct(per-face):"
                      << " bottom=" << (gxBottom > 0 ? 100.0 * gxBottomR / gxBottom : 0.0) << "%"
                      << " top=" << (gxTop > 0 ? 100.0 * gxTopR / gxTop : 0.0) << "%"
                      << " sideX=" << (gxSideX > 0 ? 100.0 * gxSideXR / gxSideX : 0.0) << "%"
                      << " sideY=" << (gxSideY > 0 ? 100.0 * gxSideYR / gxSideY : 0.0) << "%"
                      << std::endl;
            std::cout << "[MOP]   GlassExitSideXSplit: +X=" << gxPlusX
                      << " -X=" << gxMinusX << std::endl;

            // Expert SideX-TIR z-distribution (2026-04-22):
            // slots 40-49: SideX face TIR 광자의 photon.position.z 분포
            // bin0=[-100,-90], bin1=[-90,-80], ..., bin9=[-10,0]
            // 주의: HACK reject 발생 시에도 동일 슬롯에 누적됨 (HACK 분기 안에서)
            uint64_t sideXTIRtotal = 0;
            for (int b = 0; b < 10; b++) sideXTIRtotal += dc[40 + b];
            uint32_t hackRejectCnt = dc[60];
            std::cout << "[MOP] === Expert SideX-TIR z Distribution ===" << std::endl;
            std::cout << "[MOP]   SideXTIRtotal=" << sideXTIRtotal
                      << " (HACKreject=" << hackRejectCnt << ")" << std::endl;

            // [SideXDiag 2026-04-27] SideX boundary TIR vs refract + cosI histogram
            uint32_t sideXTIRcount = dc[70];
            uint32_t sideXRefractCount = dc[71];
            uint32_t sideXTotalDecisions = sideXTIRcount + sideXRefractCount;
            std::cout << "[SideXDiag] SideX boundary decisions: TIR=" << sideXTIRcount
                      << " refract=" << sideXRefractCount
                      << " (TIR fraction=" << (sideXTotalDecisions > 0 ?
                          100.0 * sideXTIRcount / sideXTotalDecisions : 0) << "%)" << std::endl;
            std::cout << "[SideXDiag]   cosI histogram (critical cosI for n=1.58→1.0 = "
                      << cos(asin(1.0/1.58)) << " ≈ 0.774 — refract if cosI>0.774, TIR if cosI<0.774):" << std::endl;
            for (int b = 0; b < 10; b++) {
                double lo = b * 0.1, hi = lo + 0.1;
                double pct = (sideXTotalDecisions > 0) ? 100.0 * dc[4000+b] / sideXTotalDecisions : 0;
                std::cout << "[SideXDiag]   cosI=[" << lo << "," << hi << "]  count="
                          << dc[4000+b] << "  (" << pct << "%)" << std::endl;
            }
            // stepDist log10 histogram
            std::cout << "[SideXDiag]   stepDist log10 histogram (mm):" << std::endl;
            const char* sd_labels[11] = {
                "<1e-9",   "1e-9..1e-8",  "1e-8..1e-7",  "1e-7..1e-6",
                "1e-6..1e-5", "1e-5..1e-4", "1e-4..1e-3", "1e-3..1e-2",
                "1e-2..1e-1", "1e-1..1.0", ">1.0"};
            for (int b = 0; b < 11; b++) {
                double pct = (sideXTotalDecisions > 0) ? 100.0 * dc[4010+b] / sideXTotalDecisions : 0;
                std::cout << "[SideXDiag]   step=" << sd_labels[b]
                          << "  count=" << dc[4010+b] << "  (" << pct << "%)" << std::endl;
            }
            std::cout << "[MOP]   z-bins(10mm each, z range [-100,0]):" << std::endl;
            for (int b = 0; b < 10; b++) {
                int zlo = -100 + 10 * b;
                int zhi = zlo + 10;
                double pct = (sideXTIRtotal > 0) ? (100.0 * dc[40 + b] / sideXTIRtotal) : 0.0;
                std::cout << "[MOP]     bin" << b
                          << " z=[" << zlo << "," << zhi << "] "
                          << "count=" << dc[40 + b]
                          << " (" << pct << "%)" << std::endl;
            }

            // ============================================================
            // Expert SideX-hit-pos (2026-04-22):
            //   slot 100 = total SideX hit count seen (>= dump capacity 100)
            //   slot 200..1099 = (x,y,z,dx,dy,dz,nx,ny,nz) raw float bits per entry
            //   slot 1100..1199 = meta: flags|TIR<<8|matIn<<16|matOut<<24
            // 분석 목적:
            //   (a) x 가 ±100mm 정확히 인지 vs drift (BVH float32 corner artifact?)
            //   (b) z 가 corner edge (0 또는 -100) 근처인지 vs face mid
            //   (c) corner (|x|≈100 + |z|≈0 or 100) 와 face-mid 분포 비율
            //   (d) GPU SideX hit 이 정상 multi-bounce 인지 vs spurious BVH
            // ============================================================
            uint32_t sideXHitTotal = dc[100];
            uint32_t sideXDumpCnt  = std::min<uint32_t>(sideXHitTotal, 100u);
            std::cout << "[MOP] === Expert SideX-hit-pos dump ==="
                      << " totalSeen=" << sideXHitTotal
                      << " dumped=" << sideXDumpCnt << std::endl;
            std::cout << "[MOP]   columns: idx, x, y, z, dx, dy, dz, nx, ny, nz, flags, TIR, matIn, matOut" << std::endl;
            // 분포 통계
            int cnt_xCorner = 0;     // |x| > 99.99
            int cnt_xDrift  = 0;     // 99.5 < |x| < 99.99 (BVH drift suspect)
            int cnt_xMid    = 0;     // |x| <= 99.5 (face mid)
            int cnt_zTopEdge= 0;     // z > -1 (top corner edge near z=0)
            int cnt_zBotEdge= 0;     // z < -99 (bottom corner edge near z=-100)
            int cnt_zMid    = 0;     // -99 <= z <= -1
            int cnt_TIR     = 0;
            int cnt_multi   = 0;
            for (uint32_t i = 0; i < sideXDumpCnt; i++) {
                uint32_t base = 200u + i * 9u;
                float fx, fy, fz, fdx, fdy, fdz, fnx, fny, fnz;
                std::memcpy(&fx,  &dc[base + 0], 4);
                std::memcpy(&fy,  &dc[base + 1], 4);
                std::memcpy(&fz,  &dc[base + 2], 4);
                std::memcpy(&fdx, &dc[base + 3], 4);
                std::memcpy(&fdy, &dc[base + 4], 4);
                std::memcpy(&fdz, &dc[base + 5], 4);
                std::memcpy(&fnx, &dc[base + 6], 4);
                std::memcpy(&fny, &dc[base + 7], 4);
                std::memcpy(&fnz, &dc[base + 8], 4);
                uint32_t meta = dc[1100 + i];
                uint32_t flags  = (meta >>  0) & 0xFFu;
                uint32_t isTIR  = (meta >>  8) & 0xFFu;
                uint32_t matIn  = (meta >> 16) & 0xFFu;
                uint32_t matOut = (meta >> 24) & 0xFFu;
                std::cout << "[MOP-PT] " << i
                          << " " << fx << " " << fy << " " << fz
                          << " " << fdx << " " << fdy << " " << fdz
                          << " " << fnx << " " << fny << " " << fnz
                          << " " << flags << " " << isTIR
                          << " " << matIn << " " << matOut << std::endl;
                float ax = std::fabs(fx);
                if (ax > 99.99f)      cnt_xCorner++;
                else if (ax > 99.5f)  cnt_xDrift++;
                else                  cnt_xMid++;
                if (fz > -1.0f)         cnt_zTopEdge++;
                else if (fz < -99.0f)   cnt_zBotEdge++;
                else                    cnt_zMid++;
                if (isTIR) cnt_TIR++;
                if (flags >= 1) cnt_multi++;
            }
            std::cout << "[MOP] === SideX-hit-pos summary (dumped " << sideXDumpCnt << ") ==="
                      << std::endl;
            std::cout << "[MOP]   x: |x|>99.99 (corner) = " << cnt_xCorner
                      << " | 99.5<|x|<99.99 (drift) = " << cnt_xDrift
                      << " | |x|<=99.5 (mid) = " << cnt_xMid << std::endl;
            std::cout << "[MOP]   z: top-edge (z>-1) = " << cnt_zTopEdge
                      << " | bot-edge (z<-99) = " << cnt_zBotEdge
                      << " | mid = " << cnt_zMid << std::endl;
            std::cout << "[MOP]   TIR=" << cnt_TIR
                      << " multiBounce=" << cnt_multi << std::endl;
        }

        // 2026-05-03: V11/V12/V15/V16 verbose dumps gated by logLevel (per-Propagate
        // 65만줄 출력으로 wall time 4-5× 증가). production 에서는 비활성.
        if (m_logLevel >= 2) {
        // === 검증 11: 모든 dielectric boundary 의 R 값 직접 dump ===
        {
            uint32_t* dc = (uint32_t*)[m_diagBuffer contents];
            uint32_t bndDumpTotal = dc[1200];
            uint32_t bndDumpCnt   = std::min<uint32_t>(bndDumpTotal, 100u);
            std::cout << "[MOP] === V11 boundary R dump === totalSeen=" << bndDumpTotal
                      << " dumped=" << bndDumpCnt << std::endl;
            std::cout << "[MOP]   columns: idx, cosI, n1, n2, R, polX, polY, polZ, Rs" << std::endl;
            for (uint32_t i = 0; i < bndDumpCnt; i++) {
                uint32_t base = 1300u + i * 8u;
                float cosI = *reinterpret_cast<float*>(&dc[base+0]);
                float n1   = *reinterpret_cast<float*>(&dc[base+1]);
                float n2   = *reinterpret_cast<float*>(&dc[base+2]);
                float R    = *reinterpret_cast<float*>(&dc[base+3]);
                float pX   = *reinterpret_cast<float*>(&dc[base+4]);
                float pY   = *reinterpret_cast<float*>(&dc[base+5]);
                float pZ   = *reinterpret_cast<float*>(&dc[base+6]);
                float Rs   = *reinterpret_cast<float*>(&dc[base+7]);
                std::cout << "[MOP-V11] " << i << " "
                          << cosI << " " << n1 << " " << n2 << " " << R << " "
                          << pX << " " << pY << " " << pZ << " " << Rs << std::endl;
            }
        }
        // === 검증 12: Glass bottom reflect 후 새 dir/pos/pol dump ===
        {
            uint32_t* dc = (uint32_t*)[m_diagBuffer contents];
            uint32_t v12Total = dc[1500];
            uint32_t v12Cnt = std::min<uint32_t>(v12Total, 100u);
            std::cout << "[MOP] === V12 Glass-bottom reflect (post) dump === totalSeen=" << v12Total
                      << " dumped=" << v12Cnt << std::endl;
            std::cout << "[MOP]   columns: idx, dirX, dirY, dirZ, posX, posY, posZ, polX, polY, polZ" << std::endl;
            for (uint32_t i = 0; i < v12Cnt; i++) {
                uint32_t base = 1600u + i * 9u;
                float dx = *reinterpret_cast<float*>(&dc[base+0]);
                float dy = *reinterpret_cast<float*>(&dc[base+1]);
                float dz = *reinterpret_cast<float*>(&dc[base+2]);
                float px = *reinterpret_cast<float*>(&dc[base+3]);
                float py = *reinterpret_cast<float*>(&dc[base+4]);
                float pz = *reinterpret_cast<float*>(&dc[base+5]);
                float lx = *reinterpret_cast<float*>(&dc[base+6]);
                float ly = *reinterpret_cast<float*>(&dc[base+7]);
                float lz = *reinterpret_cast<float*>(&dc[base+8]);
                std::cout << "[MOP-V12] " << i << " "
                          << dx << " " << dy << " " << dz << " "
                          << px << " " << py << " " << pz << " "
                          << lx << " " << ly << " " << lz << std::endl;
            }
        }
        // === V16-GPU: 광자별 boundary attempts hist ===
        if (m_photonBuffer) {
            MOPPhoton* photons = (MOPPhoton*)[m_photonBuffer contents];
            uint32_t pcount = m_primaryPhotonCount.load(std::memory_order_relaxed);
            static std::atomic<uint64_t> s_v16Hist[32] = {};
            for (uint32_t i = 0; i < pcount; i++) {
                uint32_t att = (photons[i].flags >> 27) & 0x1Fu;
                s_v16Hist[att]++;
            }
            static std::atomic<uint64_t> s_v16PhotonTotal{0};
            s_v16PhotonTotal += pcount;
            uint64_t totalSeen = s_v16PhotonTotal.load();
            if (pcount > 0) {
                std::cout << "[MOP-V16] cumulative photons=" << totalSeen << " hist:";
                uint64_t total = 0; uint64_t weighted = 0;
                for (int i = 0; i < 32; i++) {
                    uint64_t v = s_v16Hist[i].load();
                    if (v > 0) std::cout << " [" << i << "]=" << v;
                    total += v; weighted += v * i;
                }
                std::cout << " mean=" << (total > 0 ? (double)weighted/total : 0.0) << std::endl;
            }
        }
        // === V15-GPU: Air→Glass entry attempts/reflect 누적 ===
        {
            uint32_t* dc = (uint32_t*)[m_diagBuffer contents];
            static std::atomic<uint64_t> s_v15Att{0};
            static std::atomic<uint64_t> s_v15Refl{0};
            s_v15Att += dc[4000];
            s_v15Refl += dc[4001];
            uint64_t att = s_v15Att.load();
            uint64_t refl = s_v15Refl.load();
            std::cout << "[MOP-V15] Air→Glass entry cumulative: attempts=" << att
                      << " reflect=" << refl
                      << " rate=" << (att > 0 ? 100.0 * refl / att : 0.0) << "%"
                      << std::endl;
        }
        }  // m_logLevel >= 1
        // === 검증 13: 실제 (rand, R) 페어 dump === (logLevel >= 1 only — verbose)
        if (m_logLevel >= 2) {
            uint32_t* dc = (uint32_t*)[m_diagBuffer contents];
            uint32_t v13Total = dc[2500];
            uint32_t v13Cnt = std::min<uint32_t>(v13Total, 200u);
            std::cout << "[MOP] === V13 (rand, R) pair dump === totalSeen=" << v13Total
                      << " dumped=" << v13Cnt << std::endl;
            std::cout << "[MOP]   columns: idx, cosI, R, rand_peek, will_reflect_flag" << std::endl;
            uint32_t v13_reflect = 0, v13_refract = 0;
            double v13_rand_sum = 0.0, v13_R_sum = 0.0;
            for (uint32_t i = 0; i < v13Cnt; i++) {
                uint32_t base = 2600u + i * 4u;
                float cosI = *reinterpret_cast<float*>(&dc[base+0]);
                float R    = *reinterpret_cast<float*>(&dc[base+1]);
                float rnd  = *reinterpret_cast<float*>(&dc[base+2]);
                uint32_t flag = dc[base+3];
                if (flag) v13_reflect++; else v13_refract++;
                v13_rand_sum += rnd; v13_R_sum += R;
                std::cout << "[MOP-V13] " << i << " "
                          << cosI << " " << R << " " << rnd << " " << flag << std::endl;
            }
            if (v13Cnt > 0) {
                std::cout << "[MOP] V13 summary: reflect=" << v13_reflect
                          << " refract=" << v13_refract
                          << " mean_R=" << (v13_R_sum/v13Cnt)
                          << " mean_rand=" << (v13_rand_sum/v13Cnt)
                          << std::endl;
            }
        }

        // Fix 61: 궤적 덤프 출력 (다중 광자, LogLevel >= 1)
        // 다중 광자 dump → photonIdx 별 그룹화 → first reflect 광자 식별 → trajectory + path 합 출력
        {
            uint32_t trajCount = *((uint32_t*)[m_trajectoryCountBuffer contents]);
            if (trajCount > 0 && m_logLevel >= 3) {
                // 2026-04-23 옵션 2 v4: 첫 batch만 dump (EventBatchSize=300 가정 — 한 번에 모든 photon)
                static bool trajPrinted = false;
                if (!trajPrinted) {
                    trajPrinted = true;
                    MOPTrajectoryStep* traj = (MOPTrajectoryStep*)[m_trajectoryBuffer contents];
                    uint32_t totalSteps = std::min(trajCount, (uint32_t)40000);

                    // photonIdx 별 step 인덱스 그룹화 + path 합 계산
                    std::map<uint32_t, std::vector<uint32_t>> photonSteps;
                    std::map<uint32_t, double> photonPathSum;
                    std::map<uint32_t, int> photonReflectStep;  // 첫 reflect step 인덱스
                    for (uint32_t i = 0; i < totalSteps; i++) {
                        uint32_t pid = traj[i].photonIdx;
                        photonSteps[pid].push_back(i);
                        photonPathSum[pid] += traj[i].stepDist;
                        // 진짜 reflect: 다른 물질 경계에서 outcome=reflect/TIR
                        // (Air-Air SameMaterial 경계 — AirHalfScorer 통과 등 — 제외)
                        bool isReflect = (traj[i].processType == 0) &&
                                         (traj[i].outcome == 1 || traj[i].outcome == 2) &&
                                         (traj[i].matIdIn != traj[i].matIdOut);
                        if (isReflect && photonReflectStep.find(pid) == photonReflectStep.end()) {
                            photonReflectStep[pid] = (int)i;
                        }
                    }

                    std::cout << "[Fix61] Total trajectory steps recorded = " << totalSteps
                              << ", unique photons = " << photonSteps.size() << std::endl;

                    // First reflect 광자 = photonIdx 가 가장 작은 reflect 광자
                    uint32_t firstReflectIdx = UINT32_MAX;
                    for (auto& kv : photonReflectStep) {
                        if (kv.first < firstReflectIdx) firstReflectIdx = kv.first;
                    }

                    // 광자별 step 합 요약 출력 (처음 30개)
                    std::cout << "[Fix61] === Per-photon step path summary (first 30 photons) ===" << std::endl;
                    int summaryCount = 0;
                    for (auto& kv : photonSteps) {
                        if (summaryCount++ >= 30) break;
                        bool hasReflect = photonReflectStep.count(kv.first) > 0;
                        std::cout << "[Fix61] photon " << kv.first
                                  << " : " << kv.second.size() << " steps, sumPath="
                                  << photonPathSum[kv.first] << " mm"
                                  << (hasReflect ? " [REFLECT]" : "") << std::endl;
                    }

                    if (firstReflectIdx == UINT32_MAX) {
                        std::cout << "[Fix61] No reflect photon found in dump buffer." << std::endl;
                    } else {
                        std::cout << "[Fix61] === First reflect photon = #" << firstReflectIdx
                                  << " ; total steps=" << photonSteps[firstReflectIdx].size()
                                  << " ; sumPath=" << photonPathSum[firstReflectIdx] << " mm ==="
                                  << std::endl;
                        std::cout << "[Fix61] step | pos(x,y,z) | dir(x,y,z) | stepDist | process | matIn->Out | n1->n2 | cosI | R | outcome | triId | volId" << std::endl;

                        const auto& stepIdx = photonSteps[firstReflectIdx];
                        for (size_t k = 0; k < stepIdx.size(); k++) {
                            const MOPTrajectoryStep& s = traj[stepIdx[k]];
                            const char* procName = "?";
                            if (s.processType == 0) procName = "BOUNDARY";
                            else if (s.processType == 1) procName = "ABSORB";
                            else if (s.processType == 2) procName = "RAYLEIGH";
                            else if (s.processType == 3) procName = "MIE";

                            const char* outName = "?";
                            if (s.outcome == 0) outName = "transmit";
                            else if (s.outcome == 1) outName = "reflect";
                            else if (s.outcome == 2) outName = "TIR";
                            else if (s.outcome == 3) outName = "bndAbs";
                            else if (s.outcome == 4) outName = "bulkAbs";

                            printf("[Fix61] %3zu | (%8.3f,%8.3f,%8.3f) | (%6.3f,%6.3f,%6.3f) | %8.4f | %-8s | %d->%d | %.4f->%.4f | %.4f | %.4f | %-8s | %d | %d\n",
                                   k, s.posX, s.posY, s.posZ,
                                   s.dirX, s.dirY, s.dirZ,
                                   s.stepDist, procName,
                                   s.matIdIn, s.matIdOut,
                                   s.n1, s.n2, s.cosI, s.fresnelR, outName,
                                   s.triangleId, s.volumeId);
                        }
                        std::cout << "[Fix61] === End of First Reflect Trajectory ===" << std::endl;
                    }

                    // 2026-04-23 옵션 2 v3: photon 0~9999 모두 dump
                    std::cout << "[Fix61-ALL] === All photons (idx 0~9999) trajectory dump ===" << std::endl;
                    std::cout << "[Fix61-ALL] track | step | pos(x,y,z) | dir(x,y,z) | stepDist | process | matIn->Out | cosI | R | outcome" << std::endl;
                    for (uint32_t pid = 0; pid < 10000; pid++) {
                        if (photonSteps.find(pid) == photonSteps.end()) continue;
                        const auto& pidx = photonSteps[pid];
                        for (size_t k = 0; k < pidx.size(); k++) {
                            const MOPTrajectoryStep& s = traj[pidx[k]];
                            const char* procName = "?";
                            if (s.processType == 0) procName = "BOUNDARY";
                            else if (s.processType == 1) procName = "ABSORB";
                            const char* outName = "?";
                            if (s.outcome == 0) outName = "transmit";
                            else if (s.outcome == 1) outName = "reflect";
                            else if (s.outcome == 2) outName = "TIR";
                            else if (s.outcome == 3) outName = "bndAbs";
                            printf("[Fix61-ALL] t%3u | %2zu | (%12.8f,%12.8f,%12.8f) | (%12.9f,%12.9f,%12.9f) | n=(%12.9f,%12.9f,%12.9f) | %8.4f | %-8s | %d->%d | %.4f | %.4f | %-8s\n",
                                   pid, k, s.posX, s.posY, s.posZ,
                                   s.dirX, s.dirY, s.dirZ,
                                   s.normX, s.normY, s.normZ,
                                   s.stepDist, procName,
                                   s.matIdIn, s.matIdOut,
                                   s.cosI, s.fresnelR, outName);
                        }
                    }
                    std::cout << "[Fix61-ALL] === End ===" << std::endl;
                }
            }
        }
        }  // 2026-05-03: outer m_logLevel >= 1 (전체 진단 블록 close)

        return m_lastHitCount;
    }
}

// Fix 71: GPU 작업 완료 대기 및 결과 수확
void MetalOpticalEngineImpl::WaitForGPU() {
    if (m_hasPendingWork && m_pendingCmdBuf) {
        [m_pendingCmdBuf waitUntilCompleted];

        // 결과 읽기
        uint32_t hitCount = *((uint32_t*)[m_hitCountBuffer contents]);
        m_lastHitCount = std::min(hitCount, m_hitBufferCapacity);

        uint32_t transitCount = *((uint32_t*)[m_transitCountBuffer contents]);
        m_lastTransitCount = std::min(transitCount, m_transitBufferCapacity);
        // 2026-05-03 fix B: overflow detect — shader 가 cap 초과 transit 생성 시 GPU
        // 측에서 cap 까지만 기록하고 transitCount 는 실제 시도한 수를 보관 (atomic counter).
        // overflow 면 (1) 다음 dispatch 의 cap 2x 확장 요청 (2) 이번 dispatch 결과는
        // truncation 되어 result invalid → fatal abort + 사용자에게 batch 줄이거나 retry 안내.
        if (transitCount > m_transitBufferCapacity) {
            std::cerr << "\n[MOP][FATAL] Transit hit buffer overflow: shader generated "
                      << transitCount << " > cap " << m_transitBufferCapacity << "\n"
                      << "  Result silently truncated → physics WRONG. Aborting.\n"
                      << "  Workaround: smaller EventBatchSize OR rebuild with larger initial cap.\n"
                      << "  Auto-grow: next process run will use 2x cap (" << m_transitBufferCapacity*2 << ")\n"
                      << std::endl;
            m_transitGrowRequested = true;
            std::abort();  // 정확한 결과 보장이 우선 — 잘린 데이터 사용 금지
        }

        // 통계
        m_stats.totalPhotonsGenerated += m_pendingTotalPhotons;
        m_stats.totalPhotonsDetected += m_lastHitCount;
        m_stats.totalPhotonsPropagated += m_pendingTotalPhotons;
        m_stats.totalTransitHitsRecorded += m_lastTransitCount;

        auto completionTime = std::chrono::high_resolution_clock::now();
        double gpuMs = std::chrono::duration<double, std::milli>(completionTime - m_pendingSubmitTime).count();
        m_stats.gpuTimeMs += gpuMs;

        m_pendingCmdBuf = nil;
        m_hasPendingWork = false;
    }
}

uint32_t MetalOpticalEngineImpl::GetHits(MOPHit* hits, uint32_t maxHits) {
    WaitForGPU();  // Fix 71: 결과 필요 시점에 대기
    // GPU 버퍼 직접 접근 (Apple Silicon 통합 메모리: 복사 1회만, 중간 벡터 불필요)
    uint32_t count = std::min(m_lastHitCount, maxHits);
    if (count > 0 && m_hitBuffer) {
        MOPHit* hitData = (MOPHit*)[m_hitBuffer contents];
        memcpy(hits, hitData, count * sizeof(MOPHit));
    }
    return count;
}

bool MetalOpticalEngineImpl::IsGPUAvailable() const {
    return m_device != nil;
}

void MetalOpticalEngineImpl::GetStats(MOPPropagationStats& stats) const {
    stats = m_stats;
}

void MetalOpticalEngineImpl::SetLogLevel(int level) {
    m_logLevel = level;
}

// Fix 41: Transit hit 데이터 가져오기 (MOPHit 형식으로 변환)
uint32_t MetalOpticalEngineImpl::GetTransitHits(MOPHit* hits, uint32_t maxHits) {
    WaitForGPU();  // Fix 71: 결과 필요 시점에 대기
    uint32_t count = std::min(m_lastTransitCount, maxHits);
    if (count > 0 && m_transitHitBuffer) {
        // TransitHit → MOPHit 변환 (위치+에너지만 사용)
        // GPU의 TransitHit 구조: position(3f), direction(3f), energy(f), weight(f), photonWeight(f), _pad(f)
        struct GPUTransitHit {
            float posX, posY, posZ;
            float dirX, dirY, dirZ;
            float energy;
            float weight;        // geometric path length (mm)
            float photonWeight;  // 광자 가중치 (default 1.0)
            float _pad;
        };
        GPUTransitHit* transitData = (GPUTransitHit*)[m_transitHitBuffer contents];
        for (uint32_t i = 0; i < count; i++) {
            hits[i].posX = transitData[i].posX;
            hits[i].posY = transitData[i].posY;
            hits[i].posZ = transitData[i].posZ;
            hits[i].energy = transitData[i].energy;
            hits[i].wavelength = (transitData[i].energy > 0) ? 1239.84198f / transitData[i].energy : 0;
            hits[i].time = 0;
            hits[i].volumeId = 0; // transit marker
            hits[i].parentTrackId = 0;
            // Fix 70: transit hit 데이터를 MOPHit에 인코딩
            // flags = geometric pathLength, time = photonWeight, _padding = dirX, dirY, dirZ
            hits[i].time = transitData[i].photonWeight;  // 광자 가중치 in time field
            float pathLen = transitData[i].weight;  // geometric path only
            memcpy(&hits[i].flags, &pathLen, sizeof(float));
            float dirX = transitData[i].dirX;
            float dirY = transitData[i].dirY;
            float dirZ = transitData[i].dirZ;
            memcpy(&hits[i]._padding[0], &dirX, sizeof(float));
            memcpy(&hits[i]._padding[1], &dirY, sizeof(float));
            memcpy(&hits[i]._padding[2], &dirZ, sizeof(float));
        }
    }
    return count;
}

// SF1 (2026-04-24): GPU surface flux scorer 메서드 구현
// 5명 합의 design — bounded plane crossing detection.
uint32_t MetalOpticalEngineImpl::RegisterSurfaceScorer(const MOPSurfaceDef& def) {
    // surfaceId = scorer 등록 순번 (0-based). 개수 제한 없음 — vector 동적 확장.
    if (def.surfaceId >= m_surfaceDefs.size()) {
        m_surfaceDefs.resize(def.surfaceId + 1);
    }
    m_surfaceDefs[def.surfaceId] = def;
    m_surfaceDefs[def.surfaceId].enabled = 1;

    // numActiveSurfaces 재계산 (highest enabled index + 1)
    m_numActiveSurfaces = 0;
    for (uint32_t i = 0; i < m_surfaceDefs.size(); i++) {
        if (m_surfaceDefs[i].enabled) m_numActiveSurfaces = i + 1;
    }
    m_config.numSurfaces = m_numActiveSurfaces;

    // GPU 버퍼 sync — capacity 부족 시 동적 재할당
    if (m_surfaceDefCapacity < (uint32_t)m_surfaceDefs.size()) {
        m_surfaceDefCapacity = (uint32_t)m_surfaceDefs.size();
        m_surfaceDefBuffer = [m_device newBufferWithLength:m_surfaceDefCapacity * sizeof(MOPSurfaceDef)
                                          options:MTLResourceStorageModeShared];
    }
    if (m_surfaceDefBuffer && !m_surfaceDefs.empty()) {
        memcpy([m_surfaceDefBuffer contents], m_surfaceDefs.data(),
               m_surfaceDefs.size() * sizeof(MOPSurfaceDef));
    }
    if (m_logLevel >= 2) {
        std::cout << "[MOP-SF1] Registered surface " << def.surfaceId
                  << " origin=(" << def.originX << "," << def.originY << "," << def.originZ << ")"
                  << " normal=(" << def.normalX << "," << def.normalY << "," << def.normalZ << ")"
                  << " halfExtents=(" << def.halfExtentU << "," << def.halfExtentV << ")"
                  << " active=" << m_numActiveSurfaces << std::endl;
    }
    return def.surfaceId;
}

void MetalOpticalEngineImpl::ClearSurfaceScorers() {
    m_surfaceDefs.clear();
    m_numActiveSurfaces = 0;
    m_config.numSurfaces = 0;
    if (m_surfaceDefBuffer) {
        memset([m_surfaceDefBuffer contents], 0, m_surfaceDefCapacity * sizeof(MOPSurfaceDef));
    }
}

uint32_t MetalOpticalEngineImpl::GetSurfaceHits(MOPSurfaceHit* hits, uint32_t maxHits) {
    WaitForGPU();
    if (m_surfaceHitCountBuffer) {
        m_lastSurfaceHitCount = *((uint32_t*)[m_surfaceHitCountBuffer contents]);
    }
    uint32_t count = std::min(m_lastSurfaceHitCount, maxHits);
    if (count > 0 && m_surfaceHitBuffer) {
        memcpy(hits, [m_surfaceHitBuffer contents], count * sizeof(MOPSurfaceHit));
    }
    return count;
}

uint32_t MetalOpticalEngineImpl::GetLastSurfaceHitCount() const {
    if (m_surfaceHitCountBuffer) {
        return *((uint32_t*)[m_surfaceHitCountBuffer contents]);
    }
    return m_lastSurfaceHitCount;
}

// ============================================================
// C API 구현
// ============================================================
extern "C" {

MOPEngineHandle MOPEngine_Create(void) {
    auto* impl = new MetalOpticalEngineImpl();
    if (!impl->Initialize()) {
        delete impl;
        return nullptr;
    }
    return (MOPEngineHandle)impl;
}

void MOPEngine_Destroy(MOPEngineHandle engine) {
    if (engine) delete (MetalOpticalEngineImpl*)engine;
}

void MOPEngine_SetConfig(MOPEngineHandle engine, const MOPSimConfig* config) {
    if (engine && config) ((MetalOpticalEngineImpl*)engine)->SetConfig(*config);
}

void MOPEngine_GetConfig(MOPEngineHandle engine, MOPSimConfig* config) {
    if (engine && config) *config = ((MetalOpticalEngineImpl*)engine)->GetConfig();
}

int MOPEngine_LoadTOPASParameters(MOPEngineHandle engine, const char* filePath) {
    if (!engine || !filePath) return -1;
    return ((MetalOpticalEngineImpl*)engine)->LoadTOPASParameters(filePath);
}

// F2 정석 fix (2026-04-23): material name → id 조회 (DDA scorer material 필터용)
uint32_t MOPEngine_GetMaterialId(MOPEngineHandle engine, const char* name) {
    if (!engine || !name) return 0xFFFFFFFFu;
    return ((MetalOpticalEngineImpl*)engine)->GetMaterialIdByName(name);
}

uint32_t MOPEngine_RegisterMaterial(MOPEngineHandle engine,
                                     const MOPMaterialProperties* material,
                                     const char* name) {
    if (!engine || !material) return UINT32_MAX;
    return ((MetalOpticalEngineImpl*)engine)->RegisterMaterial(*material, name);
}

uint32_t MOPEngine_RegisterSurface(MOPEngineHandle engine,
                                    const MOPSurfaceProperties* surface,
                                    const char* name) {
    if (!engine || !surface) return UINT32_MAX;
    return ((MetalOpticalEngineImpl*)engine)->RegisterSurface(*surface, name);
}

void MetalOpticalEngineImpl::AddBoxGeometry(const MOPBoxGeometry& box) {
    m_boxGeometries.push_back(box);
    if (m_logLevel >= 1) {
        std::cout << "[MOP-G1] AddBoxGeometry vol=" << box.volumeId
                  << " HL=(" << box.HLX << "," << box.HLY << "," << box.HLZ << ")"
                  << " ctr=(" << box.centerX << "," << box.centerY << "," << box.centerZ << ")"
                  << " matIn=" << box.materialIdInside << " matOut=" << box.materialIdOutside << "\n";
    }
}

extern "C" void MOPEngine_AddBoxGeometry(MOPEngineHandle engine, const MOPBoxGeometry* box) {
    if (engine && box) ((MetalOpticalEngineImpl*)engine)->AddBoxGeometry(*box);
}

void MetalOpticalEngineImpl::AddSphereGeometry(const MOPSphereGeometry& sphere) {
    m_sphereGeometries.push_back(sphere);
    if (m_logLevel >= 1) {
        std::cout << "[MOP-Sphere] AddSphereGeometry vol=" << sphere.volumeId
                  << " ctr=(" << sphere.centerX << "," << sphere.centerY << "," << sphere.centerZ << ")"
                  << " R=" << sphere.radius
                  << " matIn=" << sphere.materialIdInside << " matOut=" << sphere.materialIdOutside << "\n";
    }
}

extern "C" void MOPEngine_AddSphereGeometry(MOPEngineHandle engine, const MOPSphereGeometry* sphere) {
    if (engine && sphere) ((MetalOpticalEngineImpl*)engine)->AddSphereGeometry(*sphere);
}

void MetalOpticalEngineImpl::AddCylinderGeometry(const MOPCylinderGeometry& cyl_in) {
    // 2026-05-17 정석 fix: 다중 cyl 의 coincident boundary (예: BarrelShell.RMin ==
    // InnerMieCyl.RMax) 시 shader 의 cyl analytic 2-cyl 처리에서 segfault. 광자가
    // boundary 정확 위에서 양 cyl 의 hit candidate 가 t≈0 → ε push 100nm 후도
    // borderline → infinite loop 또는 invalid memory access.
    // Auto-shift: coincident 발견 시 100nm push (ε push 와 동일 scale) 으로 분리.
    // shift 영향 = 100nm / R = ~2.5e-6 (fp32 ULP 작음) — 시뮬레이션 결과 영향 미미.
    MOPCylinderGeometry cyl = cyl_in;
    const float kCoincidentTol = 1e-3f;  // 1μm — coincident detection threshold
    const float kEpsShift      = 1e-4f;  // 100nm — same as boundary ε push
    int shiftCount = 0;
    for (const auto& existing : m_cylinderGeometries) {
        // Concentric check (centers ≈ same)
        if (std::abs(cyl.centerX - existing.centerX) > kCoincidentTol) continue;
        if (std::abs(cyl.centerY - existing.centerY) > kCoincidentTol) continue;
        if (std::abs(cyl.centerZ - existing.centerZ) > kCoincidentTol) continue;
        // Case 1: new cyl 의 Rmax ≈ existing 의 Rmin (new 가 existing 안에 nested)
        if (existing.innerRadius > kCoincidentTol &&
            std::abs(cyl.radius - existing.innerRadius) < kCoincidentTol) {
            cyl.radius -= kEpsShift;
            shiftCount++;
        }
        // Case 2: new cyl 의 Rmin ≈ existing 의 Rmax (existing 가 new 안에 nested)
        if (cyl.innerRadius > kCoincidentTol &&
            std::abs(cyl.innerRadius - existing.radius) < kCoincidentTol) {
            cyl.innerRadius += kEpsShift;
            shiftCount++;
        }
        // Case 3: new cyl 의 Rmin ≈ existing 의 Rmin (둘 다 inner cyl)
        if (cyl.innerRadius > kCoincidentTol && existing.innerRadius > kCoincidentTol &&
            std::abs(cyl.innerRadius - existing.innerRadius) < kCoincidentTol) {
            cyl.innerRadius += kEpsShift;
            shiftCount++;
        }
        // Case 4: new cyl 의 Rmax ≈ existing 의 Rmax (둘 다 outer cyl)
        if (std::abs(cyl.radius - existing.radius) < kCoincidentTol) {
            cyl.radius -= kEpsShift;
            shiftCount++;
        }
    }
    m_cylinderGeometries.push_back(cyl);
    if (m_logLevel >= 1) {
        std::cout << "[MOP-Cyl] AddCylinderGeometry vol=" << cyl.volumeId
                  << " ctr=(" << cyl.centerX << "," << cyl.centerY << "," << cyl.centerZ << ")"
                  << " R=" << cyl.radius << " HL=" << cyl.halfLength
                  << " matIn=" << cyl.materialIdInside << " matOut=" << cyl.materialIdOutside;
        if (shiftCount > 0) {
            std::cout << "  (auto-shifted " << shiftCount << "x by " << (kEpsShift*1e6) << "nm)";
        }
        std::cout << "\n";
    }
}

extern "C" void MOPEngine_AddCylinderGeometry(MOPEngineHandle engine, const MOPCylinderGeometry* cyl) {
    if (engine && cyl) ((MetalOpticalEngineImpl*)engine)->AddCylinderGeometry(*cyl);
}

void MetalOpticalEngineImpl::AddTorusGeometry(const MOPTorusGeometry& tor) {
    m_torusGeometries.push_back(tor);
    if (m_logLevel >= 1) {
        std::cout << "[MOP-Torus] AddTorusGeometry vol=" << tor.volumeId
                  << " ctr=(" << tor.centerX << "," << tor.centerY << "," << tor.centerZ << ")"
                  << " RTor=" << tor.rTor << " RMax=" << tor.rMax
                  << " matIn=" << tor.materialIdInside << " matOut=" << tor.materialIdOutside << "\n";
    }
}

extern "C" void MOPEngine_AddTorusGeometry(MOPEngineHandle engine, const MOPTorusGeometry* tor) {
    if (engine && tor) ((MetalOpticalEngineImpl*)engine)->AddTorusGeometry(*tor);
}

// Phase 4 (2026-05-18): G4Polyhedra analytic intersection
void MetalOpticalEngineImpl::AddPolyhedraGeometry(const MOPPolyhedraGeometry& ph) {
    m_polyhedraGeometries.push_back(ph);
    if (m_logLevel >= 1) {
        std::cout << "[MOP-Poly] AddPolyhedraGeometry vol=" << ph.volumeId
                  << " NSides=" << ph.numSides
                  << " rMax=" << ph.rMax << "mm HL=" << ph.halfLengthAxis
                  << "mm ctr=(" << ph.centerX << "," << ph.centerY << "," << ph.centerZ << ")"
                  << " axis=(" << ph.axisX << "," << ph.axisY << "," << ph.axisZ << ")"
                  << " matIn=" << ph.materialIdInside << " matOut=" << ph.materialIdOutside
                  << " surfaceId=" << ph.surfaceId
                  << std::endl;
    }
}

extern "C" void MOPEngine_AddPolyhedraGeometry(MOPEngineHandle engine, const MOPPolyhedraGeometry* ph) {
    if (engine && ph) ((MetalOpticalEngineImpl*)engine)->AddPolyhedraGeometry(*ph);
}

void MOPEngine_AddVolumeMesh(MOPEngineHandle engine,
                              const MOPTriangle* triangles,
                              uint32_t numTriangles,
                              uint32_t volumeId) {
    if (engine) ((MetalOpticalEngineImpl*)engine)->AddVolumeMesh(triangles, numTriangles, volumeId);
}

int MOPEngine_BuildAccelerationStructure(MOPEngineHandle engine) {
    if (!engine) return -1;
    return ((MetalOpticalEngineImpl*)engine)->BuildAccelerationStructure();
}

void MOPEngine_SetBorderSurface(MOPEngineHandle engine,
                                 uint32_t v1, uint32_t v2, uint32_t surfId) {
    if (engine) ((MetalOpticalEngineImpl*)engine)->SetBorderSurface(v1, v2, surfId);
}

void MOPEngine_SetSkinSurface(MOPEngineHandle engine, uint32_t vol, uint32_t surfId) {
    if (engine) ((MetalOpticalEngineImpl*)engine)->SetSkinSurface(vol, surfId);
}

void MOPEngine_AddGenstep(MOPEngineHandle engine, const MOPGenstep* genstep) {
    if (engine && genstep) ((MetalOpticalEngineImpl*)engine)->AddGenstep(*genstep);
}

uint32_t MOPEngine_Propagate(MOPEngineHandle engine) {
    if (!engine) return 0;
    return ((MetalOpticalEngineImpl*)engine)->Propagate();
}

uint32_t MOPEngine_GetHits(MOPEngineHandle engine, MOPHit* hits, uint32_t maxHits) {
    if (!engine || !hits) return 0;
    return ((MetalOpticalEngineImpl*)engine)->GetHits(hits, maxHits);
}

void MOPEngine_ResetGensteps(MOPEngineHandle engine) {
    if (engine) ((MetalOpticalEngineImpl*)engine)->ResetGensteps();
}

void MOPEngine_AddPrimaryPhoton(MOPEngineHandle engine, const MOPPhoton* photon) {
    if (engine && photon) ((MetalOpticalEngineImpl*)engine)->AddPrimaryPhoton(*photon);
}

void MOPEngine_ResetPrimaryPhotons(MOPEngineHandle engine) {
    if (engine) ((MetalOpticalEngineImpl*)engine)->ResetPrimaryPhotons();
}

uint32_t MOPEngine_GetPrimaryPhotonCount(MOPEngineHandle engine) {
    if (!engine) return 0;
    return ((MetalOpticalEngineImpl*)engine)->GetPrimaryPhotonCount();
}

int MOPEngine_IsGPUAvailable(MOPEngineHandle engine) {
    if (!engine) return 0;
    return ((MetalOpticalEngineImpl*)engine)->IsGPUAvailable() ? 1 : 0;
}

void MOPEngine_GetStats(MOPEngineHandle engine, MOPPropagationStats* stats) {
    if (engine && stats) ((MetalOpticalEngineImpl*)engine)->GetStats(*stats);
}

void MOPEngine_SetLogLevel(MOPEngineHandle engine, int level) {
    if (engine) ((MetalOpticalEngineImpl*)engine)->SetLogLevel(level);
}

// 2026-05-12: TOPAS Ts/Seed 를 GPU RNG base 로 전달
void MOPEngine_SetBaseSeed(MOPEngineHandle engine, uint32_t seed) {
    if (engine) ((MetalOpticalEngineImpl*)engine)->SetBaseSeed(seed);
}

void MOPEngine_SetWorldSize(MOPEngineHandle engine, float x, float y, float z) {
    if (engine) ((MetalOpticalEngineImpl*)engine)->SetWorldSize(x, y, z);
}

uint32_t MOPEngine_GetTransitHits(MOPEngineHandle engine, MOPHit* hits, uint32_t maxHits) {
    if (!engine || !hits) return 0;
    return ((MetalOpticalEngineImpl*)engine)->GetTransitHits(hits, maxHits);
}

uint32_t MOPEngine_GetLastTransitCount(MOPEngineHandle engine) {
    if (!engine) return 0;
    return ((MetalOpticalEngineImpl*)engine)->GetLastTransitCount();
}

void MOPEngine_SetScoringAABB(MOPEngineHandle engine,
                               float minX, float minY, float minZ,
                               float maxX, float maxY, float maxZ) {
    if (engine) ((MetalOpticalEngineImpl*)engine)->SetScoringAABB(minX, minY, minZ, maxX, maxY, maxZ);
}

// SF1 (2026-04-24): GPU surface flux scorer C API
uint32_t MOPEngine_RegisterSurfaceScorer(MOPEngineHandle engine,
                                          const MOPSurfaceDef* def) {
    if (!engine || !def) return 0xFFFFFFFFu;
    return ((MetalOpticalEngineImpl*)engine)->RegisterSurfaceScorer(*def);
}

void MOPEngine_ClearSurfaceScorers(MOPEngineHandle engine) {
    if (engine) ((MetalOpticalEngineImpl*)engine)->ClearSurfaceScorers();
}

uint32_t MOPEngine_GetSurfaceHits(MOPEngineHandle engine,
                                   MOPSurfaceHit* hits, uint32_t maxHits) {
    if (!engine || !hits) return 0;
    return ((MetalOpticalEngineImpl*)engine)->GetSurfaceHits(hits, maxHits);
}

uint32_t MOPEngine_GetLastSurfaceHitCount(MOPEngineHandle engine) {
    if (!engine) return 0;
    return ((MetalOpticalEngineImpl*)engine)->GetLastSurfaceHitCount();
}

void MOPEngine_RunGPUDDA(MOPEngineHandle engine,
                         float tx, float ty, float tz,
                         float fx, float fy, float fz,
                         uint32_t nx, uint32_t ny, uint32_t nz) {
    if (engine) ((MetalOpticalEngineImpl*)engine)->RunGPUDDA(tx,ty,tz,fx,fy,fz,nx,ny,nz, 0xFFFFFFFFu);
}

void MOPEngine_RunGPUDDA_WithMaterial(MOPEngineHandle engine,
                                       float tx, float ty, float tz,
                                       float fx, float fy, float fz,
                                       uint32_t nx, uint32_t ny, uint32_t nz,
                                       uint32_t scoringMaterialId) {
    if (engine) ((MetalOpticalEngineImpl*)engine)->RunGPUDDA(tx,ty,tz,fx,fy,fz,nx,ny,nz, scoringMaterialId);
}

const float* MOPEngine_GetDDABinBuffer(MOPEngineHandle engine) {
    if (!engine) return nullptr;
    return ((MetalOpticalEngineImpl*)engine)->GetDDABinBuffer();
}

uint32_t MOPEngine_GetDDABinCount(MOPEngineHandle engine) {
    if (!engine) return 0;
    return ((MetalOpticalEngineImpl*)engine)->GetDDABinCount();
}

float MOPEngine_GetDDAFluenceScale(MOPEngineHandle engine) {
    if (!engine) return 1.0f;
    return ((MetalOpticalEngineImpl*)engine)->GetDDAFluenceScale();
}

// Multi-scorer race fix (2026-05-01): per-scorer accumulating buffer C API.
void MOPEngine_SetScorerEnergyFilter(MOPEngineHandle engine, int handle,
                                      float energyLow, float energyHigh) {
    if (!engine) return;
    ((MetalOpticalEngineImpl*)engine)->SetScorerEnergyFilter(handle, energyLow, energyHigh);
}

int MOPEngine_RegisterScorerBinBuffer(MOPEngineHandle engine, uint32_t totalBins,
                                      uint32_t nBinsE, float eMinEv, float eMaxEv) {
    if (!engine) return -1;
    return ((MetalOpticalEngineImpl*)engine)->RegisterScorerBinBuffer(totalBins, nBinsE, eMinEv, eMaxEv);
}

extern "C" void MOPEngine_RunGPUDDA_AccumulateExt(MOPEngineHandle engine, int scorerHandle,
                                                    float tx, float ty, float tz,
                                                    float fx, float fy, float fz,
                                                    uint32_t nx, uint32_t ny, uint32_t nz,
                                                    uint32_t scoringMaterialId,
                                                    uint32_t voxelType,
                                                    float rMin, float phiStart, float thetaStart)
{
    if (engine) ((MetalOpticalEngineImpl*)engine)->RunGPUDDA_AccumulateExt(
        scorerHandle, tx, ty, tz, fx, fy, fz, nx, ny, nz,
        scoringMaterialId, voxelType, rMin, phiStart, thetaStart);
}

void MOPEngine_RunGPUDDA_Accumulate(MOPEngineHandle engine, int scorerHandle,
                                     float tx, float ty, float tz,
                                     float fx, float fy, float fz,
                                     uint32_t nx, uint32_t ny, uint32_t nz,
                                     uint32_t scoringMaterialId)
{
    if (engine) ((MetalOpticalEngineImpl*)engine)->RunGPUDDA_Accumulate(
        scorerHandle, tx, ty, tz, fx, fy, fz, nx, ny, nz, scoringMaterialId);
}

const float* MOPEngine_GetScorerBinBuffer(MOPEngineHandle engine, int scorerHandle,
                                              uint32_t* outCount) {
    if (!engine) {
        if (outCount) *outCount = 0;
        return nullptr;
    }
    return ((MetalOpticalEngineImpl*)engine)->GetScorerBinBuffer(scorerHandle, outCount);
}

void MOPEngine_ResetScorerBinBuffers(MOPEngineHandle engine) {
    if (engine) ((MetalOpticalEngineImpl*)engine)->ResetScorerBinBuffers();
}

} // extern "C"

// ============================================================
// Fix 65: GPU DDA 스코어링 구현
// ============================================================
void MetalOpticalEngineImpl::RunGPUDDA(
    float compTransX, float compTransY, float compTransZ,
    float compFullX, float compFullY, float compFullZ,
    uint32_t nBinsX, uint32_t nBinsY, uint32_t nBinsZ,
    uint32_t scoringMaterialId)
{
    if (!m_ddaPipeline || !m_transitHitBuffer) return;

    uint32_t transitCount = m_lastTransitCount;
    if (transitCount == 0) return;

    uint32_t totalBins = nBinsX * nBinsY * nBinsZ;
    if (totalBins == 0) return;

    // Fix 78측정-2: 이전 DDA 미완료 시 대기 (CPU memset이 GPU writes와 race 방지)
    WaitForDDA();

    // 빈 버퍼 할당/확장 (grow-only, 할당 시에만 0 초기화)
    // 2026-05-12 fix: uint32 → atomic_float (4 byte 동일, overflow 없음)
    if (totalBins > m_ddaBinCount || !m_ddaBinBuffer) {
        m_ddaBinBuffer = [m_device newBufferWithLength:totalBins * sizeof(float)
                                   options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked];
        m_ddaBinCount = totalBins;
        if (m_logLevel >= 2) {
            std::cout << "[MOP] Fix 65: DDA bin buffer allocated: "
                      << totalBins << " bins (" << (totalBins * 4 / 1024 / 1024) << " MB, atomic_float)" << std::endl;
        }
    }
    // Fix 78측정-2: bin 초기화는 GPU blit으로 — cmdBuf 안에서 dispatch 전에 실행되어 큐 순서 보장

    // DDA config 설정
    float binSizeX = compFullX / (float)nBinsX;
    float binSizeY = compFullY / (float)nBinsY;
    float binSizeZ = compFullZ / (float)nBinsZ;
    float binVolume = binSizeX * binSizeY * binSizeZ;

    MOPDDAConfig ddaConfig;
    ddaConfig.compTransX = compTransX;
    ddaConfig.compTransY = compTransY;
    ddaConfig.compTransZ = compTransZ;
    ddaConfig.compFullX = compFullX;
    ddaConfig.compFullY = compFullY;
    ddaConfig.compFullZ = compFullZ;
    ddaConfig.nBinsX = nBinsX;
    ddaConfig.nBinsY = nBinsY;
    ddaConfig.nBinsZ = nBinsZ;
    ddaConfig.totalTransitHits = transitCount;
    ddaConfig.binVolume = binVolume;
    ddaConfig.fluenceScale = 1.0f;  // 2026-05-12: atomic_float 누적 (uint32 overflow fix), scaling 불필요
    ddaConfig.scoringMaterialId = scoringMaterialId;  // F2 정석 fix
    // Phase 2.1+ (2026-05-16): legacy RunGPUDDA 는 BOX voxel.
    ddaConfig.voxelType = 0u;
    ddaConfig.rMin = 0.0f;
    ddaConfig.phiStart = 0.0f;
    ddaConfig.thetaStart = 0.0f;

    memcpy([m_ddaConfigBuffer contents], &ddaConfig, sizeof(MOPDDAConfig));

    // GPU DDA 커널 dispatch
    @autoreleasepool {
        id<MTLCommandBuffer> cmdBuf = [m_commandQueue commandBuffer];

        // Fix 78측정-2: GPU blit으로 bin 버퍼 초기화 (CPU memset 대체)
        // → 큐 순서 보장 + CPU 동기화 불필요
        {
            id<MTLBlitCommandEncoder> blitEnc = [cmdBuf blitCommandEncoder];
            [blitEnc fillBuffer:m_ddaBinBuffer
                          range:NSMakeRange(0, totalBins * sizeof(float))
                          value:0];
            [blitEnc endEncoding];
        }

        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];

        [encoder setComputePipelineState:m_ddaPipeline];
        [encoder setBuffer:m_transitHitBuffer    offset:0 atIndex:0];
        [encoder setBuffer:m_ddaConfigBuffer     offset:0 atIndex:1];
        [encoder setBuffer:m_ddaBinBuffer        offset:0 atIndex:2];
        [encoder setBuffer:m_saturationFlagBuffer offset:0 atIndex:3];  // 2026-05-15 saturation 감지

        NSUInteger threadWidth = [m_ddaPipeline threadExecutionWidth];
        [encoder dispatchThreads:MTLSizeMake(transitCount, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(threadWidth, 1, 1)];
        [encoder endEncoding];

        // Fix 78측정: DDA dispatch GPU 시간 캡처
        MetalOpticalEngineImpl* self = this;
        [cmdBuf addCompletedHandler:^(id<MTLCommandBuffer> cb) {
            double ms = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
            self->m_stats.gpuDDAMs += ms;
            self->m_stats.ddaDispatchCount++;
        }];
        [cmdBuf commit];
        // Fix 78측정-2: waitUntilCompleted 제거 — GetDDABinBuffer()에서 lazy 대기
        m_pendingDDACmdBuf = cmdBuf;
    }
}

void MetalOpticalEngineImpl::WaitForDDA() {
    if (m_pendingDDACmdBuf) {
        [m_pendingDDACmdBuf waitUntilCompleted];
        m_pendingDDACmdBuf = nil;
    }
}

const float* MetalOpticalEngineImpl::GetDDABinBuffer() {
    // Fix 78측정-2: 지연 동기화 — 미완료 DDA cmdBuf 대기 후 반환
    // 2026-05-12: uint32 → atomic_float (overflow fix)
    WaitForDDA();
    // 2026-05-15: fp32 ULP lost-value 비율 기반 check (legacy path).
    if (m_saturationFlagBuffer && m_ddaBinBuffer) {
        float lostVal = *(float*)[m_saturationFlagBuffer contents];
        const float* bins = (const float*)[m_ddaBinBuffer contents];
        double total = 0.0;
        for (uint32_t i = 0; i < m_ddaBinCount; i++) total += (double)bins[i];
        double frac = (total > 0.0) ? (double)lostVal / total : 0.0;
        if (frac > 0.01) {
            std::cerr << "\n========================================================\n"
                      << "[FATAL] GPUOpticalPhotonFluence: fp32 ULP saturation detected (DDA legacy)\n"
                      << "  lost-value = " << lostVal << " / bin-total = " << total
                      << " → " << (frac * 100.0) << "% (> 1% threshold)\n"
                      << "  Workaround: bin 더 쪼개세요 (각 dim 늘리거나 부피 작게).\n"
                      << "  참조: memory [[u5scan_gpu_singlebin_fp32_saturation_2026_05_15]]\n"
                      << "========================================================\n"
                      << std::endl;
            std::abort();
        }
    }
    if (!m_ddaBinBuffer) return nullptr;
    return (const float*)[m_ddaBinBuffer contents];
}

// Multi-scorer race fix (2026-05-01): per-scorer persistent buffer.
void MetalOpticalEngineImpl::SetScorerEnergyFilter(int handle, float energyLow, float energyHigh) {
    if (handle < 0 || (size_t)handle >= m_scorerBuffers.size()) return;
    m_scorerBuffers[handle].energyLow  = energyLow;
    m_scorerBuffers[handle].energyHigh = energyHigh;
    if (m_logLevel >= 2) {
        std::cout << "[TsGPU-Fluence] Scorer handle=" << handle
                  << " energy filter set: [" << energyLow << ", " << energyHigh << "] eV" << std::endl;
    }
}

int MetalOpticalEngineImpl::RegisterScorerBinBuffer(uint32_t totalBins, uint32_t nBinsE,
                                                    float eMinEv, float eMaxEv) {
    if (!m_device || totalBins == 0) return -1;
    if (nBinsE < 1) nBinsE = 1;
    @autoreleasepool {
        ScorerBinBuffer sb;
        sb.totalBins = totalBins;   // voxel count (X*Y*Z)
        sb.nBinsE = nBinsE;
        sb.eMinEv = eMinEv;
        sb.eMaxEv = eMaxEv;
        // uint64 emulation (dual-uint32 with carry): 1B focal-spot fp32 overflow 방지.
        //   full buffer = totalBins(voxel) × nBinsE × 2 uint32 (low, high).
        uint32_t fullBins = totalBins * nBinsE;
        sb.buffer = [m_device newBufferWithLength:fullBins * 2 * sizeof(uint32_t)
                                          options:MTLResourceStorageModeShared];
        if (!sb.buffer) return -1;
        memset([sb.buffer contents], 0, fullBins * 2 * sizeof(uint32_t));
        // per-scorer DDAConfig buffer (CPU writes per dispatch — 공유 시 race)
        // 2026-05-12 fix: HazardTrackingModeUntracked 제거
        sb.configBuffer = [m_device newBufferWithLength:sizeof(MOPDDAConfig)
                                                options:MTLResourceStorageModeShared];
        if (!sb.configBuffer) return -1;
        sb.pendingCmdBuf = nil;
        m_scorerBuffers.push_back(sb);
        return (int)(m_scorerBuffers.size() - 1);
    }
}

void MetalOpticalEngineImpl::RunGPUDDA_Accumulate(
    int scorerHandle,
    float compTransX, float compTransY, float compTransZ,
    float compFullX, float compFullY, float compFullZ,
    uint32_t nBinsX, uint32_t nBinsY, uint32_t nBinsZ,
    uint32_t scoringMaterialId)
{
    // Legacy 호환: BOX voxel type 으로 위임.
    RunGPUDDA_AccumulateExt(scorerHandle,
                             compTransX, compTransY, compTransZ,
                             compFullX, compFullY, compFullZ,
                             nBinsX, nBinsY, nBinsZ,
                             scoringMaterialId,
                             /*voxelType=*/0u,    // BOX
                             /*rMin=*/0.0f,
                             /*phiStart=*/0.0f,
                             /*thetaStart=*/0.0f);
}

void MetalOpticalEngineImpl::RunGPUDDA_AccumulateExt(
    int scorerHandle,
    float compTransX, float compTransY, float compTransZ,
    float compFullX, float compFullY, float compFullZ,
    uint32_t nBinsX, uint32_t nBinsY, uint32_t nBinsZ,
    uint32_t scoringMaterialId,
    uint32_t voxelType,
    float rMin, float phiStart, float thetaStart)
{
    if (scorerHandle < 0 || (size_t)scorerHandle >= m_scorerBuffers.size()) return;
    if (!m_ddaPipeline || !m_transitHitBuffer) return;

    uint32_t transitCount = m_lastTransitCount;
    if (transitCount == 0) return;

    uint32_t totalBins = nBinsX * nBinsY * nBinsZ;
    if (totalBins == 0) return;

    ScorerBinBuffer& sb = m_scorerBuffers[scorerHandle];
    if (totalBins != sb.totalBins) return;

    // per-scorer pending wait (이 scorer 만 직렬화, 다른 scorer 와 무관).
    if (sb.pendingCmdBuf) {
        [sb.pendingCmdBuf waitUntilCompleted];
        sb.pendingCmdBuf = nil;
    }

    // DDA config — voxelType 별 bin volume 계산.
    // BOX:      compFullX/Y/Z = dx*dy*dz product
    // CYLINDER: bin 마다 volume 다름 (annular sector) — shader 측 정확 계산.
    //           binVolume field 는 대표값 (전체/총bin) 으로 sanity 만.
    // SPHERE:   같음 — bin 마다 다름, shader 측 계산.
    float binVolume = 0.0f;
    if (voxelType == 0u) {
        float binSizeX = compFullX / (float)nBinsX;
        float binSizeY = compFullY / (float)nBinsY;
        float binSizeZ = compFullZ / (float)nBinsZ;
        binVolume = binSizeX * binSizeY * binSizeZ;
    } else if (voxelType == 1u) {
        // CYLINDER total volume = π * (RMax² - RMin²) * (2HL) * (DPhi/2π)
        float rMax = rMin + compFullX;
        float dphi = compFullY;  // (rad)
        float hZ   = compFullZ;
        float totV = (float)M_PI * (rMax*rMax - rMin*rMin) * hZ * (dphi / (2.0f*(float)M_PI));
        binVolume = totV / (float)totalBins;
    } else { // SPHERE
        float rMax = rMin + compFullX;
        float dphi = compFullZ;
        float dtheta = compFullY;
        // (4π/3)(rMax³ - rMin³) * (dtheta/π) * (dphi/2π) — approximation
        float totV = (4.0f/3.0f) * (float)M_PI * (rMax*rMax*rMax - rMin*rMin*rMin)
                   * (dtheta / (float)M_PI) * (dphi / (2.0f*(float)M_PI));
        binVolume = totV / (float)totalBins;
    }

    MOPDDAConfig ddaConfig;
    ddaConfig.compTransX = compTransX;
    ddaConfig.compTransY = compTransY;
    ddaConfig.compTransZ = compTransZ;
    ddaConfig.compFullX = compFullX;
    ddaConfig.compFullY = compFullY;
    ddaConfig.compFullZ = compFullZ;
    ddaConfig.nBinsX = nBinsX;
    ddaConfig.nBinsY = nBinsY;
    ddaConfig.nBinsZ = nBinsZ;
    ddaConfig.totalTransitHits = transitCount;
    ddaConfig.binVolume = binVolume;
    ddaConfig.fluenceScale = 1.0f;  // 2026-05-12: atomic_float 누적 (uint32 overflow fix), scaling 불필요
    ddaConfig.scoringMaterialId = scoringMaterialId;
    ddaConfig.voxelType = voxelType;
    ddaConfig.rMin = rMin;
    ddaConfig.phiStart = phiStart;
    ddaConfig.thetaStart = thetaStart;
    // 2026-05-19 multi-scorer photon energy filter (eV).
    ddaConfig.energyLow  = sb.energyLow;
    ddaConfig.energyHigh = sb.energyHigh;
    // 2026-05-20: energy/wavelength binning. nBinsE=1 (default) → kernel eBinOffset=0.
    ddaConfig.nBinsE = sb.nBinsE;
    ddaConfig.eMinEv = sb.eMinEv;
    ddaConfig.eMaxEv = sb.eMaxEv;
    ddaConfig._pad[0] = 0u;
    ddaConfig._pad[1] = 0u;

    memcpy([sb.configBuffer contents], &ddaConfig, sizeof(MOPDDAConfig));

    @autoreleasepool {
        id<MTLCommandBuffer> cmdBuf = [m_commandQueue commandBuffer];

        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        [encoder setComputePipelineState:m_ddaPipeline];
        [encoder setBuffer:m_transitHitBuffer     offset:0 atIndex:0];
        [encoder setBuffer:sb.configBuffer        offset:0 atIndex:1];
        [encoder setBuffer:sb.buffer              offset:0 atIndex:2];
        [encoder setBuffer:m_saturationFlagBuffer offset:0 atIndex:3];  // 2026-05-15 saturation 감지

        NSUInteger threadWidth = [m_ddaPipeline threadExecutionWidth];
        [encoder dispatchThreads:MTLSizeMake(transitCount, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(threadWidth, 1, 1)];
        [encoder endEncoding];

        MetalOpticalEngineImpl* self = this;
        [cmdBuf addCompletedHandler:^(id<MTLCommandBuffer> cb) {
            double ms = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
            self->m_stats.gpuDDAMs += ms;
            self->m_stats.ddaDispatchCount++;
        }];
        [cmdBuf commit];
        sb.pendingCmdBuf = cmdBuf;  // per-scorer pending tracker
    }
}

const float* MetalOpticalEngineImpl::GetScorerBinBuffer(int scorerHandle, uint32_t* outCount) {
    if (scorerHandle < 0 || (size_t)scorerHandle >= m_scorerBuffers.size()) {
        if (outCount) *outCount = 0;
        return nullptr;
    }
    ScorerBinBuffer& sb = m_scorerBuffers[scorerHandle];
    // per-scorer pending wait (이 scorer 의 마지막 dispatch 만)
    if (sb.pendingCmdBuf) {
        [sb.pendingCmdBuf waitUntilCompleted];
        sb.pendingCmdBuf = nil;
    }
    // 2026-05-15: fp32 ULP lost-value 비율 기반 abort.
    //   shader 가 lost contrib 의 값을 lostValueSum 에 atomic_float 누적. host 는
    //   binBuffer 총합 대비 비율로 판정 (>1% = systematic bias → abort).
    //   tiny corner-clip lost 는 합산해도 작아 통과. typical-size lost 는 비율 큼.
    // uint64 emulation reconstruction: binBuffer[bin*2]=low, [bin*2+1]=high. value=(high<<32|low)/SCALE.
    const uint32_t* uintBuf = (const uint32_t*)[sb.buffer contents];
    uint32_t fullBins = sb.totalBins * sb.nBinsE;
    sb.reconstructedView.assign(fullBins, 0.0f);
    constexpr double FLUENCE_QUANT_SCALE = 1e10;  // shader 와 동기화 (1e10)
    double total = 0.0;
    for (uint32_t i = 0; i < fullBins; i++) {
        uint64_t v = ((uint64_t)uintBuf[i*2 + 1] << 32) | (uint64_t)uintBuf[i*2];
        double val = (double)v / FLUENCE_QUANT_SCALE;
        sb.reconstructedView[i] = (float)val;
        total += val;
    }

    if (m_saturationFlagBuffer) {
        float lostVal = *(float*)[m_saturationFlagBuffer contents];
        double frac = (total > 0.0) ? (double)lostVal / total : 0.0;
        if (frac > 0.01) {
            std::cerr << "[FATAL] GPUOpticalPhotonFluence: quantize-floor lost "
                      << (frac * 100.0) << "% (>1%). SCALE=" << FLUENCE_QUANT_SCALE << std::endl;
            std::abort();
        }
    }
    if (outCount) *outCount = fullBins;  // 2026-05-20: full size (voxel × nBinsE)
    return sb.reconstructedView.data();
}

void MetalOpticalEngineImpl::ResetScorerBinBuffers() {
    WaitForDDA();
    for (auto& sb : m_scorerBuffers) {
        if (sb.buffer && sb.totalBins > 0) {
            // 2026-05-17 fix: uint64 emulation. 2026-05-20: full = totalBins × nBinsE.
            uint32_t fullBins = sb.totalBins * sb.nBinsE;
            memset([sb.buffer contents], 0, fullBins * 2 * sizeof(uint32_t));
        }
    }
}

