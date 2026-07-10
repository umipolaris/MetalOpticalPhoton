/**
 * TsGPUOpticalPhysics.hh
 * TOPAS Extension: GPU 가속 광학 물리 모듈
 *
 * Geant4의 G4Scintillation/G4Cerenkov 프로세스를 오버라이드하여
 * 광학 광자 생성 정보(genstep)를 수집하고, Metal GPU 엔진으로 전달
 *
 * TOPAS 파라미터 파일에서 사용:
 *   sv:Ph/Default/Modules = 1 "g4optical"
 *   → 이 모듈이 g4optical을 대체하며 GPU 가속을 제공
 *
 * 또는 추가 모듈로 사용:
 *   s:Ph/Default/Modules = 2 "g4optical" "TsGPUOpticalPhysics"
 *   b:Ph/Default/TsGPUOpticalPhysics/Active = "True"
 */

#ifndef TS_GPU_OPTICAL_PHYSICS_HH
#define TS_GPU_OPTICAL_PHYSICS_HH

#include "MetalOpticalEngine.hh"

// Geant4 헤더
#include "G4UserSteppingAction.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4OpticalPhoton.hh"
#include "G4Scintillation.hh"
#include "G4Cerenkov.hh"
#include "G4OpBoundaryProcess.hh"
#include "G4RunManager.hh"
#include "G4EventManager.hh"

#include <vector>
#include <string>
#include <map>
#include <atomic>
#include <mutex>
#include <utility>
#include <functional>  // 2026-05-12: scorer harvest callback

class G4LogicalVolume;
class G4VPhysicalVolume;
class TsParameterManager;

// ============================================================
// GPU Optical Physics 모듈 for TOPAS
// ============================================================
class TsGPUOpticalPhysics {
public:
    /**
     * 생성자
     * @param topasParameterFile TOPAS 파라미터 파일 경로
     *        (광학 물질/표면 속성을 자동으로 파싱)
     */
    TsGPUOpticalPhysics(const std::string& topasParameterFile = "");
    ~TsGPUOpticalPhysics();

    /**
     * 초기화 - TOPAS Extension 초기화 시 호출
     * Metal GPU 엔진을 초기화하고 TOPAS 파라미터를 로드
     */
    bool Initialize();

    /**
     * 지오메트리 구축 완료 후 호출
     * Geant4 지오메트리를 삼각형 메시로 변환하여 GPU에 업로드
     */
    void BuildGeometry();

    // ============================================================
    // Geant4 UserAction 인터페이스
    // ============================================================

    /**
     * SteppingAction에서 호출
     * 신틸레이션/체렌코프 프로세스의 genstep을 수집
     */
    void ProcessStep(const G4Step* step);

    /**
     * EventAction BeginOfEvent에서 호출
     * genstep 버퍼 초기화
     */
    void BeginOfEvent();

    /**
     * EventAction EndOfEvent에서 호출
     * 누적된 genstep을 GPU로 전파하고 히트를 수집
     */
    uint32_t EndOfEvent();

    /**
     * GPU에서 검출된 히트 데이터 가져오기
     */
    uint32_t GetHits(MOPHit* hits, uint32_t maxHits);

    /**
     * GPU 통계 정보
     */
    void GetStats(MOPPropagationStats* stats);

    // ============================================================
    // TOPAS 파라미터 연동
    // ============================================================

    /**
     * TOPAS 파라미터 파일에서 광학 속성 로드
     * Ma/, Su/, Ge/ 파라미터를 파싱하여 GPU 엔진에 등록
     */
    int LoadTOPASParameters(const std::string& filePath);

    /**
     * 추가 TOPAS 파라미터 파일 로드 (include 파일 등)
     */
    int LoadAdditionalParameters(const std::string& filePath);

    /**
     * GPU 가속 활성화/비활성화
     * 비활성화 시 기본 Geant4 광학 시뮬레이션으로 폴백
     */
    void SetGPUEnabled(bool enabled) { m_gpuEnabled = enabled; }
    bool IsGPUEnabled() const { return m_gpuEnabled; }
    MOPEngineHandle GetEngine() const { return m_engine; }

    /**
     * genstep 배치 임계값 설정
     * 이 수 이상의 광자가 누적되면 GPU 전파 실행
     */
    void SetBatchThreshold(uint32_t threshold) { m_batchThreshold = threshold; }

    /**
     * Fix 63: 이벤트 배치 크기 설정 (사용자 명시 — 자동 튜닝 비활성화)
     * N개 이벤트의 genstep을 누적 후 1회 GPU dispatch → Metal 오버헤드 N분의 1
     */
    void SetEventBatchSize(int n) {
        m_batchSize = std::max(1, n);
        m_batchExplicit = true;  // Fix 78측정-3: 사용자 명시 → 자동 튜닝 무시
    }

    /**
     * Fix 78측정-3: 시나리오 자동 감지 batch 튜닝
     * 사용자가 명시 batch를 설정하지 않았으면, 스코어러의 bin 수에 따라 sweet spot 적용
     *  - totalBins ≤ 10000  : DDA 비활성 → batch=5 (per-photon 14ns)
     *  - totalBins > 10000  : DDA 활성   → batch=25 (gap/cost 균형, 100K 고해상도 -39%)
     * 측정 근거: 10K/100K 고해상도 batch sweep (5/25/50/100/200/500/1000)
     */
    void MaybeAutoTuneBatch(uint64_t totalBins) {
        // ★ Fix #19: AutoTune 비활성. m_batchSize default (10000) 유지.
        // Apple Silicon Metal driver heap leak per cmdBuf commit 우회.
        (void)totalBins;
        return;
    }
    int GetEventBatchSize() const { return m_batchSize; }
    bool IsBatchExplicit() const { return m_batchExplicit; }

    // 2026-05-12: GPU beam source mode — Module 측에서 1회 호출.
    // 첫 G4 event 의 EndOfEvent 직전 BEAM genstep 을 m_batchGensteps 에 push.
    // 2026-05-15: numPhotons int64 (지원 100B+, MOPGenstep.numPhotons uint32 한계 우회)
    void SetBeamGenstep(double cx, double cy, double cz,
                        double dx, double dy, double dz,
                        double energy_eV,
                        double posCutoffX, double posCutoffY,
                        double sigmaX, double sigmaY,
                        double cutoffX, double cutoffY,
                        int64_t numPhotons,
                        // 2026-05-17: BeamPositionCutoffShape + Spread + Polarization
                        int     posShape   = 0,     // 0=Rectangle, 1=Ellipse
                        double  posSpreadX = 0.0,   // Gaussian sigma X (mm); 0=Flat
                        double  posSpreadY = 0.0,   // Gaussian sigma Y (mm); 0=Flat
                        int     polMode    = 0,     // 0=random transverse, 1=fixed
                        double  polX       = 0.0,
                        double  polY       = 0.0,
                        double  polZ       = 0.0,
                        // 2026-05-17 추가
                        int     posDist    = 1,     // 0=None, 1=Flat, 2=Gaussian
                        int     angDist    = 0,     // 0=None, 1=Flat, 2=Gaussian
                        double  energySpreadEv = 0.0,  // Gaussian σ (eV)
                        double  timeSpread     = 0.0,  // Gaussian σ (ns)
                        double  timeCutoff     = 0.0); // abs cutoff (ns); 0=no clip
    void SetBeamMaterialId(uint32_t mid) { m_beamGenstep.materialId = mid; }
    bool IsBeamMode() const { return m_beamMode; }

    // 2026-05-17: 다중 beam 추가 — 첫 beam (SetBeamGenstep) 후 추가 beam 등록.
    // 모든 추가 beam 은 같은 SimConfig.beamPos/Ang/Pol/Spectrum 셋팅 (현재 단일 글로벌)
    // 을 공유. 향후 per-beam config 필요 시 별도 데이터.
    void AddBeamGenstep(double cx, double cy, double cz,
                        double dx, double dy, double dz,
                        double energy_eV,
                        double posCutoffX, double posCutoffY,
                        double sigmaX, double sigmaY,
                        double cutoffX, double cutoffY,
                        int64_t numPhotons);

    // Multi-scorer race fix (2026-05-01): pending Propagate cmdBuf 가 in flight 인
    // 동안 다른 scorer 가 SetScoringAABB / RunGPUDDA 호출 → engine GPU buffer race
    // → Apple Metal heap corruption. Scorer hook 시작 시 호출하여 GPU idle 보장.
    void WaitForPendingGPU();

    // Multi-scorer race fix (2026-05-01): scorer 의 DDA dispatch 를 m_gpuMutex 안에서
    // 실행. 같은 cmdQueue 에 worker thread 의 Propagate 가 동시 dispatch 되는 것을
    // 막고 transit hit buffer / DDA bin buffer race 차단. dispatch 후 동기 wait 까지.
    void RunGPUDDAWithMaterialLocked(float compTransX, float compTransY, float compTransZ,
                                      float compFullX, float compFullY, float compFullZ,
                                      uint32_t nBinsX, uint32_t nBinsY, uint32_t nBinsZ,
                                      uint32_t scoringMaterialId);

    // Multi-scorer race fix (2026-05-01) — per-scorer accumulating buffer wrappers.
    // Engine 의 m_gpuMutex 안에서 호출하여 worker thread Propagate dispatch 와 직렬화.
    int RegisterScorerBinBufferLocked(uint32_t totalBins, uint32_t nBinsE = 1,
                                      float eMinEv = 0.0f, float eMaxEv = 0.0f);
    void SetScorerEnergyFilterLocked(int handle, float eLow, float eHigh);
    void RunGPUDDAAccumulateLocked(int scorerHandle,
                                    float compTransX, float compTransY, float compTransZ,
                                    float compFullX, float compFullY, float compFullZ,
                                    uint32_t nBinsX, uint32_t nBinsY, uint32_t nBinsZ,
                                    uint32_t scoringMaterialId);
    // Phase 2.1+ (2026-05-16): cylindrical/spherical voxelization ext.
    void RunGPUDDAAccumulateExtLocked(int scorerHandle,
                                       float compTransX, float compTransY, float compTransZ,
                                       float compFullX, float compFullY, float compFullZ,
                                       uint32_t nBinsX, uint32_t nBinsY, uint32_t nBinsZ,
                                       uint32_t scoringMaterialId,
                                       uint32_t voxelType,
                                       float rMin, float phiStart, float thetaStart);
    // 2026-05-12: uint32 → atomic_float (per-voxel cumulative overflow fix)
    const float* GetScorerBinBufferLocked(int scorerHandle, uint32_t* outCount);

    // 2026-05-12: scorer 가 harvest 직후 DDA dispatch 할 수 있도록 callback 등록.
    // HarvestPendingGPUResults 가 cache 갱신 후 등록된 callback 들을 호출.
    // 다음 propagation 이 dispatch 되기 전에 DDA cmdBuf 가 queue 에 들어가
    // FIFO 순서로 올바른 batch 의 m_transitHitBuffer 데이터를 읽음.
    // race 메커니즘: 외부 harvest → callback (DDA dispatch) → propagation dispatch.
    using HarvestCallback = std::function<void()>;
    void RegisterHarvestCallback(HarvestCallback cb);

    void SetConfig(const MOPSimConfig& config);
    MOPSimConfig GetConfig() const;

    /**
     * 디버그 로그 레벨
     */
    void SetLogLevel(int level);
    int GetLogLevel() const { return m_logLevel; }

    // 2026-05-12: TOPAS Ts/Seed 를 GPU RNG base 로 전파
    void SetBaseSeed(uint32_t seed);

    // TOPAS extension param 직접 접근 (Surface ForceReflectivity 등).
    // BuildGeometry 시점에 b:Su/<name>/ForceReflectivity 같은 우리 확장
    // 파라미터를 query 하려면 TsParameterManager 필요.
    void SetParameterManager(TsParameterManager* pM) { fPm = pM; }

    /**
     * 마지막 이벤트의 히트 수 반환 (EndOfEvent 호출 후 유효)
     */
    uint32_t GetLastHitCount() const { return t_eventState.lastHitCount; }
    const std::vector<MOPHit>& GetCachedHits() const { return t_eventState.cachedHits; }

    // Fix 41: Transit hit API (스코어링 볼륨 통과 기록)
    const std::vector<MOPHit>& GetCachedTransitHits() const { return t_eventState.cachedTransitHits; }

    /**
     * 볼륨 이름으로 volumeId 조회
     * @return volumeId, 찾지 못하면 -1
     */
    int FindVolumeIdByName(const std::string& volumeName) const;

    /**
     * Fix 41b: 스코어링 AABB 설정 (매 스텝마다 위치 체크)
     * 볼륨 경계가 아닌 위치 기반으로 통과 감지 (Air-Air 경계 문제 해결)
     */
    void SetScoringAABB(float minX, float minY, float minZ,
                        float maxX, float maxY, float maxZ);

    /**
     * SF1 (2026-04-24): GPU surface flux scorer 등록
     * bounded plane을 통과하는 광자를 SurfaceHit으로 기록.
     * @param def MOPSurfaceDef (origin/normal/axisU/axisV/extents)
     * @return surfaceId 또는 0xFFFFFFFF on error
     */
    uint32_t RegisterSurfaceScorer(const MOPSurfaceDef& def);

    /**
     * SF1: 모든 surface scorer 비활성화
     */
    void ClearSurfaceScorers();

    /**
     * SF1: 누적된 SurfaceHit 캐시 접근 (현재 batch만)
     */
    const std::vector<MOPSurfaceHit>& GetCachedSurfaceHits() const {
        return t_eventState.cachedSurfaceHits;
    }

    /**
     * SF1 deficit fix (2026-04-24): Run-lifetime SurfaceHit accumulator.
     * 매 HarvestPendingGPUResults가 새 batch hits를 APPEND. EndOfRun에서 한번만 읽음.
     * cacheVer race / multi-EndOfRun 문제 회피.
     */
    const std::vector<MOPSurfaceHit>& GetAccumulatedSurfaceHits() const {
        return s_accumulatedSurfaceHits;
    }
    void ResetAccumulatedSurfaceHits() { s_accumulatedSurfaceHits.clear(); }
    static std::vector<MOPSurfaceHit> s_accumulatedSurfaceHits;

    /**
     * 2026-05-19: Run-lifetime TransitHit accumulator (SurfaceHit pattern).
     * 매 HarvestPendingGPUResults가 새 batch transit hits를 APPEND. PhaseSpace
     * scorer가 EndOfRun에서 1회 read (NumberOfHistoriesInRun 어느 값이어도 작동).
     * cachedTransitHits 는 batch boundary 마다 cleared → run-wide dump 불가능 해결.
     */
    const std::vector<MOPHit>& GetAccumulatedTransitHits() const {
        return s_accumulatedTransitHits;
    }
    void ResetAccumulatedTransitHits() { s_accumulatedTransitHits.clear(); }
    static std::vector<MOPHit> s_accumulatedTransitHits;
    // 2026-05-23: PhaseSpace scorer 가 실제 등록될 때만 transit hit 을 run-lifetime 누적.
    // 그 외(예: DDA Fluence-only 1B run)는 chunk 200개 × 수M hit 무한 누적 → host 벡터
    // realloc 시 OOM(SIGKILL). PhaseSpace 생성자가 true 로 set (없으면 누적 skip).
    static bool s_transitAccumulationEnabled;

    /**
     * Fix 42: 강제 재전파 — AABB 설정 후 이미 전파된 이벤트를 재전파
     * 스코어러 순서에 의해 AABB 없이 전파된 경우 사용
     */
    uint32_t ForceRepropagate();

    /**
     * Fix C1: Run 종료 시 잔여 배치 flush + 결과 수확 (public 래퍼)
     * 스코어러의 UserHookForEndOfRun에서 호출 → N%BatchSize≠0 이벤트의 hit 소실 방지.
     * 호출 후 GetCachedTransitHits()/GetCachedHits()에 최종 배치 결과가 반영됨.
     * @return 이번 flush에서 수확한 hit 수
     */
    uint32_t FinalizePendingBatch();

private:
    // Genstep 수집 헬퍼
    void CollectScintillationGenstep(const G4Step* step);
    void CollectCerenkovGenstep(const G4Step* step);

public:
    // Primary opticalphoton (Beam source) 직접 캡처 — public이므로 외부에서
    // (예: TsGPUOpticalKillProcess::PostStepDoIt) 호출 가능.
    //
    // 우선순위: track 인자가 비-NULL이면 그것을 사용, 아니면 step->GetTrack().
    // step만 있어도 PreStep을 통해 위치/방향/편광 추출. 둘 다 NULL이면 no-op.
    void CollectPrimaryPhoton(const G4Step* step, const G4Track* track = nullptr);
private:

    // 지오메트리 변환 (G4VSolid → 삼각형 메시)
    void ConvertGeometryToMesh();

    // Geant4 물질 → MOP materialId 매핑
    uint32_t GetMOPMaterialId(const G4Material* g4mat) const;

    // Metal 엔진 핸들
    MOPEngineHandle m_engine;

    // 상태
    bool m_initialized;
    bool m_gpuEnabled;
    bool m_geometryBuilt;
    uint32_t m_batchThreshold;
    int m_logLevel;
    // 스레드별 이벤트 상태 (MT 안전)
    struct ThreadLocalEventState {
        uint32_t lastHitCount = 0;
        bool eventAlreadyProcessed = false;
        std::vector<MOPHit> cachedHits;        // GPU 검출 히트 캐시
        std::vector<MOPHit> cachedTransitHits;  // Fix 41: 스코어링 볼륨 통과 히트 캐시
        std::vector<MOPSurfaceHit> cachedSurfaceHits;  // SF1: surface flux 통과 캐시

        // Bug A fix (2026-04-22): cache 버전 카운터.
        // HarvestPendingGPUResults 또는 ForceRepropagate 가 cache 를 갱신할 때마다
        // 증가. Scorer 는 자신의 m_lastProcessedCacheVersion 과 비교하여 같은
        // batch 데이터를 N번 재처리하지 않도록 함. 이 카운터가 없을 때는
        // BeginOfEvent 호출이 추가되면 stale cache 를 매 이벤트 재처리하여
        // 결과가 underrun 또는 overrun 됨.
        uint64_t cacheVersion = 0;
    };
    static thread_local ThreadLocalEventState t_eventState;

public:
    // Bug A fix: scorer 가 cache 처리 중복 방지를 위해 사용.
    uint64_t GetCacheVersion() const { return t_eventState.cacheVersion; }
private:

    // TOPAS 파라미터 파일 경로
    std::string m_parameterFile;

    // TsParameterManager (Module 에서 setter 로 주입). Surface 등록 시
    // b:Su/<name>/ForceReflectivity 같은 확장 파라미터 query 용.
    TsParameterManager* fPm = nullptr;

    // Fix 39: 스레드별 genstep 수집 (MT 안전)
    // 각 스레드가 자신의 genstep 벡터에 수집, EndOfEvent에서 GPU로 일괄 전송
    struct ThreadLocalGensteps {
        std::vector<MOPGenstep> gensteps;
        uint32_t pendingPhotons = 0;
        bool sent = false;  // Fix 42b: genstep이 엔진에 전송되었는지 여부 (이중 전파 방지)
    };
    static thread_local ThreadLocalGensteps t_localGensteps;

    // Geant4 물질 이름 → MOP material ID 매핑
    std::map<std::string, uint32_t> m_materialMap;

    // Geant4 표면 이름 → MOP surface ID 매핑
    std::map<std::string, uint32_t> m_surfaceMap;

    // C6: skin surface 바인딩 (논리볼륨 → surfaceId)
    std::map<G4LogicalVolume*, uint32_t> m_skinSurfaceLogVols;

    // C6: border surface 바인딩 (물리볼륨 쌍 → surfaceId)
    std::map<std::pair<G4VPhysicalVolume*, G4VPhysicalVolume*>, uint32_t> m_borderSurfacePVs;

    // volumeId → 물리볼륨 매핑 (스코어링용)
    std::map<uint32_t, G4VPhysicalVolume*> m_volumeMap;

    // m2: 체렌코프 최대 광자 수 (설정 가능)
    uint32_t m_cerenkovMaxPhotons = 300;

    // GPU 접근 동기화 — EndOfEvent(Propagate) 직렬화용만 (ProcessStep은 lock-free)
    // 2026-05-12: recursive — HarvestPendingGPUResults 에서 scorer callback 호출 시
    // callback 이 RunGPUDDAAccumulateLocked (m_gpuMutex 잠금) 으로 재진입 가능.
    std::recursive_mutex m_gpuMutex;

    // 2026-05-12: scorer 가 등록한 callback (harvest 직후 DDA dispatch).
    // HarvestPendingGPUResults 가 m_gpuPipelinePending=false 로 만든 후 호출.
    // callback 안에서 RunGPUDDAAccumulateLocked 호출 → recursive mutex 가 재진입 허용.
    std::vector<HarvestCallback> m_harvestCallbacks;

    // 2026-05-12: GPU beam source mode state
    bool m_beamMode = false;
    MOPGenstep m_beamGenstep = {};  // 첫 EndOfEvent 에 push 용 (per-chunk size)
    // 2026-05-15: MOPGenstep.numPhotons uint32 한계(4.29B) 우회 — 별도 int64 보관.
    int64_t m_beamPhotonCount = 0;
    // 2026-05-17: 다중 beam 지원 — m_beamGenstep/m_beamPhotonCount 는 첫 (또는 단일)
    // beam slot. 추가 beam 은 아래 벡터에 누적 후 chunking 시 각각 dispatch.
    std::vector<MOPGenstep> m_extraBeamGensteps;
    std::vector<int64_t>    m_extraBeamPhotonCounts;

    // Fix 63: 이벤트 배칭 — N개 이벤트의 genstep을 누적 후 1회 GPU dispatch
    int m_batchSize = 5;               // 2026-05-03 revert: batch=10000 default 가 fluence accumulate 동작 break (U5 z scorer 50% under). Fix 78측정-3 정상 default 복원.
    bool m_batchExplicit = false;    // Fix 78측정-3: 사용자 명시 여부 (자동 튜닝 게이트)
    int m_batchEventCount = 0;       // 현재 배치에 누적된 이벤트 수
    bool m_isFirstEventOfBatch = true;  // 2026-05-18 MT race fix: batch 의 첫 BeginOfEvent 만 ResetPrimaryPhotons 호출 (다중 reset 으로 인한 photon storage overwrite 방지)
    std::vector<MOPGenstep> m_batchGensteps;  // 누적 genstep 저장소
    uint32_t m_batchPendingPhotons = 0;
    uint32_t m_lastHitCount = 0;
    bool m_gpuPipelinePending = false;  // Fix 71: 비동기 GPU 작업 대기 중
    void FlushBatch();               // 잔여 배치 강제 처리
    void HarvestPendingGPUResults(); // Fix 71: 이전 GPU 결과 수확
};

#endif /* TS_GPU_OPTICAL_PHYSICS_HH */
