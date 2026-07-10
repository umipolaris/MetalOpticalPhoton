/**
 * MetalOpticalEngine.h
 * MetalOpticalPhoton - GPU-accelerated optical photon propagation for TOPAS/Geant4
 *
 * 순수 C API - TOPAS Extension(C++)에서 호출하기 위한 인터페이스
 * 내부적으로 Objective-C++/Metal 구현을 감쌈
 */

#ifndef METAL_OPTICAL_ENGINE_H
#define METAL_OPTICAL_ENGINE_H

#include "MOPTypes.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================
// 엔진 핸들 (불투명 포인터)
// ============================================================
typedef void* MOPEngineHandle;

// ============================================================
// 엔진 생성/소멸
// ============================================================

/**
 * Metal GPU 엔진 초기화
 * @return 엔진 핸들, 실패 시 NULL
 */
MOPEngineHandle MOPEngine_Create(void);

/**
 * 엔진 해제
 */
void MOPEngine_Destroy(MOPEngineHandle engine);

// ============================================================
// 시뮬레이션 설정
// ============================================================

/**
 * GPU 시뮬레이션 설정 지정
 */
void MOPEngine_SetConfig(MOPEngineHandle engine, const MOPSimConfig* config);
void MOPEngine_GetConfig(MOPEngineHandle engine, MOPSimConfig* config);

// ============================================================
// TOPAS 파라미터에서 물질/표면 속성 로드
// ============================================================

/**
 * TOPAS 파라미터 파일에서 직접 광학 속성 로드
 * 파일 내의 Ma/, Su/, Ge/ 파라미터를 파싱
 *
 * @param engine     엔진 핸들
 * @param filePath   TOPAS .topas 파라미터 파일 경로
 * @return 0=성공, -1=파일 열기 실패, -2=파싱 오류
 */
int MOPEngine_LoadTOPASParameters(MOPEngineHandle engine, const char* filePath);

/**
 * 물질 광학 속성 수동 등록
 * (TOPAS 파일을 사용하지 않고 직접 설정할 때)
 *
 * @param engine     엔진 핸들
 * @param material   물질 속성 구조체
 * @param name       물질 이름 (TOPAS에서의 Ma/이름)
 * @return 할당된 materialId
 */
uint32_t MOPEngine_RegisterMaterial(MOPEngineHandle engine,
                                     const MOPMaterialProperties* material,
                                     const char* name);

/**
 * 표면 광학 속성 수동 등록
 *
 * @param engine     엔진 핸들
 * @param surface    표면 속성 구조체
 * @param name       표면 이름 (TOPAS에서의 Su/이름)
 * @return 할당된 surfaceId
 */
uint32_t MOPEngine_RegisterSurface(MOPEngineHandle engine,
                                    const MOPSurfaceProperties* surface,
                                    const char* name);

// ============================================================
// 지오메트리 구성
// ============================================================

/**
 * 삼각형 메시로 볼륨 등록
 * Geant4 G4VSolid → 삼각형 변환 후 전달
 *
 * @param engine       엔진 핸들
 * @param triangles    삼각형 배열
 * @param numTriangles 삼각형 수
 * @param volumeId     볼륨 ID
 */
void MOPEngine_AddVolumeMesh(MOPEngineHandle engine,
                              const MOPTriangle* triangles,
                              uint32_t numTriangles,
                              uint32_t volumeId);

/**
 * 모든 볼륨 메시로부터 Metal Ray Tracing 가속 구조 빌드
 * 반드시 AddVolumeMesh 완료 후, 시뮬레이션 시작 전에 호출
 */
int MOPEngine_BuildAccelerationStructure(MOPEngineHandle engine);

// G1 (정석 fix, Opticks reference): G4Box analytic intersection 등록
void MOPEngine_AddBoxGeometry(MOPEngineHandle engine, const MOPBoxGeometry* box);

// Phase 1 (Opticks 정석): G4Sphere analytic Custom Intersection Function.
// Metal RT 의 BoundingBox geometry + intersection function 으로 ray-sphere
// quadratic 직접 계산 (mesh 우회). full sphere 만 지원.
void MOPEngine_AddSphereGeometry(MOPEngineHandle engine, const MOPSphereGeometry* sphere);

// Phase 2 (Opticks 정석): G4Tubs (TsCylinder) analytic intersection.
// Z-axis aligned, full phi, RMin=0 만 지원. 다른 parameter → mesh fallback.
void MOPEngine_AddCylinderGeometry(MOPEngineHandle engine, const MOPCylinderGeometry* cyl);

// Phase 3 (Opticks 정석): G4Torus analytic intersection.
// Quartic equation Ferrari method (fp32 + Newton refinement).
// Z-axis aligned, full phi, RMin=0 만 지원.
void MOPEngine_AddTorusGeometry(MOPEngineHandle engine, const MOPTorusGeometry* tor);

// Phase 4 (2026-05-18): G4Polyhedra (N-sided prism) analytic intersection.
// numSides ≥ 3, axis-aligned (worldTransform 의 local z-axis), full phi.
// MVP: straight prism (constant R between Z planes) — Num_z_planes=2 또는
// G4SPolyhedra 의 4-corner apex-cap RZ profile 자동 처리.
void MOPEngine_AddPolyhedraGeometry(MOPEngineHandle engine, const MOPPolyhedraGeometry* poly);

// ============================================================
// 볼륨 간 표면 바인딩 (TOPAS Ge/XXX/OpticalBehaviorTo/YYY)
// ============================================================

/**
 * 두 볼륨 사이의 경계 표면 지정
 * TOPAS의 s:Ge/Comp1/OpticalBehaviorTo/Comp2 = "SurfaceName" 에 대응
 */
void MOPEngine_SetBorderSurface(MOPEngineHandle engine,
                                 uint32_t volumeId1,
                                 uint32_t volumeId2,
                                 uint32_t surfaceId);

/**
 * 볼륨의 스킨 표면 지정
 * TOPAS의 s:Ge/Comp1/OpticalBehavior = "SurfaceName" 에 대응
 */
void MOPEngine_SetSkinSurface(MOPEngineHandle engine,
                                uint32_t volumeId,
                                uint32_t surfaceId);

// ============================================================
// Genstep 수집 및 GPU 전파
// ============================================================

/**
 * Geant4 SteppingAction에서 수집한 genstep 추가
 * 내부 버퍼에 누적되며, Propagate 호출 시 일괄 처리
 */
void MOPEngine_AddGenstep(MOPEngineHandle engine, const MOPGenstep* genstep);

/**
 * 현재 누적된 genstep들을 GPU에서 실행
 * 광자를 생성하고 전파한 후 히트를 수집
 *
 * @param engine 엔진 핸들
 * @return 검출된 히트 수
 */
uint32_t MOPEngine_Propagate(MOPEngineHandle engine);

/**
 * 마지막 Propagate 결과에서 히트 데이터 가져오기
 *
 * @param engine 엔진 핸들
 * @param hits   히트 배열 버퍼 (호출자가 할당)
 * @param maxHits 버퍼 크기
 * @return 실제 복사된 히트 수
 */
uint32_t MOPEngine_GetHits(MOPEngineHandle engine,
                            MOPHit* hits,
                            uint32_t maxHits);

/**
 * 내부 genstep 버퍼 초기화 (이벤트 간 리셋)
 */
void MOPEngine_ResetGensteps(MOPEngineHandle engine);

/**
 * Primary opticalphoton (Beam source) 직접 추가.
 * 누적된 primary는 다음 Propagate() 호출 시 propagatePhotons kernel로 전달됨.
 */
void MOPEngine_AddPrimaryPhoton(MOPEngineHandle engine, const MOPPhoton* photon);

/** Primary opticalphoton 버퍼 초기화 (이벤트 간 리셋). */
void MOPEngine_ResetPrimaryPhotons(MOPEngineHandle engine);

/** 현재 누적된 primary opticalphoton 수. */
uint32_t MOPEngine_GetPrimaryPhotonCount(MOPEngineHandle engine);

// ============================================================
// 상태 조회
// ============================================================

/**
 * GPU 사용 가능 여부 확인
 * @return 1=Metal GPU 사용 가능, 0=불가
 */
int MOPEngine_IsGPUAvailable(MOPEngineHandle engine);

/**
 * 마지막 전파의 통계 정보
 */
typedef struct {
    // 누적 카운터는 uint64 — 광자 수가 10^9 를 넘으면 uint32(<4.29e9)가 오버플로
    // (예: 표준 1×1×50 100K = ~8.08e9 generated → uint32 면 3.79e9 로 wrap).
    // totalPhotonsPropagated/totalTransitHitsRecorded 는 이미 uint64 였고, 표시용
    // generated/detected 도 동일 누적이므로 같이 64-bit 로 맞춘다.
    uint64_t totalPhotonsGenerated;
    uint64_t totalPhotonsAbsorbed;
    uint64_t totalPhotonsDetected;
    uint64_t totalPhotonsLost;
    uint64_t totalBoundaryInteractions;
    double   gpuTimeMs;              // GPU 실행 시간 (ms, wall clock 기반)
    double   totalTimeMs;            // 총 시간 (데이터 전송 포함)
    // Fix 78측정: HW 카운터 기반 dispatch별 GPU 시간 (GPUStartTime/GPUEndTime)
    double   gpuPropagateMs;         // propagate cmdBuf의 정확한 GPU 시간
    double   gpuDDAMs;               // DDA cmdBuf의 정확한 GPU 시간
    uint32_t propagateDispatchCount; // propagate dispatch 횟수
    uint32_t ddaDispatchCount;       // DDA dispatch 횟수
    uint64_t totalPhotonsPropagated; // propagate에 들어간 누적 광자 수
    uint64_t totalTransitHitsRecorded; // 누적 transit hit 기록 수
} MOPPropagationStats;

void MOPEngine_GetStats(MOPEngineHandle engine, MOPPropagationStats* stats);

/**
 * 디버그 로그 레벨 설정
 * 0=없음, 1=오류, 2=경고, 3=정보, 4=디버그
 */
void MOPEngine_SetLogLevel(MOPEngineHandle engine, int level);

/* 2026-05-12: TOPAS Ts/Seed → GPU base seed (mixed per batch) */
void MOPEngine_SetBaseSeed(MOPEngineHandle engine, uint32_t seed);

/**
 * 월드 크기 설정 (mm 단위)
 * Geant4 월드 볼륨 크기와 일치시켜야 함
 */
void MOPEngine_SetWorldSize(MOPEngineHandle engine, float x, float y, float z);

/**
 * Fix 41: Transit hit 데이터 가져오기
 * 스코어링 볼륨을 통과한 광자의 위치를 MOPHit 형식으로 반환
 */
uint32_t MOPEngine_GetTransitHits(MOPEngineHandle engine, MOPHit* hits, uint32_t maxHits);

/**
 * Fix 41: 마지막 전파의 transit hit 수 반환
 */
uint32_t MOPEngine_GetLastTransitCount(MOPEngineHandle engine);

/**
 * Fix 41b: 스코어링 AABB 설정
 * 매 전파 스텝에서 광자가 이 AABB 내에 있으면 transit hit으로 기록
 * 볼륨 경계 교차 없이도 (Air-Air 등) 정확하게 통과 감지
 */
void MOPEngine_SetScoringAABB(MOPEngineHandle engine,
                               float minX, float minY, float minZ,
                               float maxX, float maxY, float maxZ);

// ============================================================
// SF1 (2026-04-24): GPU surface flux scorer API
// 정석 fix — TsScoreSurfaceTrackCount의 GPU 등가물.
// 광자가 등록된 bounded plane을 통과할 때 SurfaceHit 기록.
// ============================================================

/**
 * Surface scorer 등록.
 * @param engine    엔진 핸들
 * @param def       MOPSurfaceDef (origin/normal/axisU/axisV/extents/surfaceId/enabled)
 * @return 등록된 surfaceId (scorer 등록 순번, 0-based, 개수 제한 없음)
 */
uint32_t MOPEngine_RegisterSurfaceScorer(MOPEngineHandle engine,
                                          const MOPSurfaceDef* def);

/**
 * 모든 surface scorer 비활성화 (run 사이 reset 용).
 */
void MOPEngine_ClearSurfaceScorers(MOPEngineHandle engine);

/**
 * 마지막 propagation의 SurfaceHit 데이터 가져오기.
 * @param hits      호출자 할당 버퍼
 * @param maxHits   버퍼 크기
 * @return 실제 복사된 hit 수
 */
uint32_t MOPEngine_GetSurfaceHits(MOPEngineHandle engine,
                                   MOPSurfaceHit* hits, uint32_t maxHits);

/**
 * 누적된 surface hit 수 (마지막 propagation).
 */
uint32_t MOPEngine_GetLastSurfaceHitCount(MOPEngineHandle engine);

/**
 * Fix 65: GPU DDA 스코어링 — transit hit을 GPU에서 직접 빈에 분배
 * 고해상도 빈(>1M)에서 CPU DDA 대비 큰 속도 향상
 */
void MOPEngine_RunGPUDDA(MOPEngineHandle engine,
                          float compTransX, float compTransY, float compTransZ,
                          float compFullX, float compFullY, float compFullZ,
                          uint32_t nBinsX, uint32_t nBinsY, uint32_t nBinsZ);

void MOPEngine_RunGPUDDA_WithMaterial(MOPEngineHandle engine,
                                       float compTransX, float compTransY, float compTransZ,
                                       float compFullX, float compFullY, float compFullZ,
                                       uint32_t nBinsX, uint32_t nBinsY, uint32_t nBinsZ,
                                       uint32_t scoringMaterialId);

/* 2026-05-12: uint32 → atomic_float (per-voxel cumulative overflow fix) */
const float* MOPEngine_GetDDABinBuffer(MOPEngineHandle engine);
uint32_t MOPEngine_GetDDABinCount(MOPEngineHandle engine);
float MOPEngine_GetDDAFluenceScale(MOPEngineHandle engine);

/**
 * Multi-scorer race fix (2026-05-01): per-scorer accumulating bin buffer.
 * 기존 RunGPUDDA / GetDDABinBuffer 는 매 dispatch 0-fill → CPU per-event read 필수.
 * 새 API: scorer 별 persistent buffer 등록 후 누적 dispatch → CPU 는 EndOfRun 1회 read.
 *
 * 사용 예 (per scorer instance):
 *   int handle = MOPEngine_RegisterScorerBinBuffer(engine, totalBins, nBinsE, eMinEv, eMaxEv);
 *   // per-event hook:
 *   MOPEngine_RunGPUDDA_Accumulate(engine, handle, ...);
 *   // EndOfRun:
 *   uint32_t count;
 *   const float* bins = MOPEngine_GetScorerBinBuffer(engine, handle, &count);
 *
 * 시그니처는 src/MetalOpticalEngine.mm 의 정의와 1:1 일치한다 (canonical 단일 소스).
 */
int MOPEngine_RegisterScorerBinBuffer(MOPEngineHandle engine, uint32_t totalBins,
                                      uint32_t nBinsE, float eMinEv, float eMaxEv);

void MOPEngine_RunGPUDDA_Accumulate(MOPEngineHandle engine, int scorerHandle,
                                     float tx, float ty, float tz,
                                     float fx, float fy, float fz,
                                     uint32_t nx, uint32_t ny, uint32_t nz,
                                     uint32_t scoringMaterialId);

const float* MOPEngine_GetScorerBinBuffer(MOPEngineHandle engine, int scorerHandle,
                                          uint32_t* outCount);

void MOPEngine_ResetScorerBinBuffers(MOPEngineHandle engine);

#ifdef __cplusplus
}
#endif

#endif /* METAL_OPTICAL_ENGINE_H */
