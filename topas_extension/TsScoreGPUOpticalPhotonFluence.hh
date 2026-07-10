/**
 * TsScoreGPUOpticalPhotonFluence.hh
 * TOPAS Extension Scorer: GPU 광학 광자 Fluence 빈 스코어러
 *
 * Fix 41: Transit hit 기반 — CPU Fluence 스코어러와 동일하게
 * 스코어링 볼륨을 통과하는 모든 광학 광자의 step_length/volume 측정
 *
 * TOPAS 파라미터 파일에서 사용:
 *   s:Sc/MyScorer/Quantity = "GPUOpticalPhotonFluence"
 *   s:Sc/MyScorer/Component = "SensorVolume"
 *   i:Sc/MyScorer/XBins = 20
 *   i:Sc/MyScorer/YBins = 20
 *   i:Sc/MyScorer/ZBins = 10
 *   s:Sc/MyScorer/OutputType = "csv"
 *   s:Sc/MyScorer/OutputFile = "SensorDepth"
 */

#ifndef TS_SCORE_GPU_OPTICAL_PHOTON_FLUENCE_HH
#define TS_SCORE_GPU_OPTICAL_PHOTON_FLUENCE_HH

#include "TsVBinnedScorer.hh"
#include <vector>
#include <mutex>
#include <set>
#include <map>
#include <string>

class TsGPUOpticalPhysics;

class TsScoreGPUOpticalPhotonFluence : public TsVBinnedScorer {
public:
    TsScoreGPUOpticalPhotonFluence(TsParameterManager* pM, TsMaterialManager* mM,
                                    TsGeometryManager* gM, TsScoringManager* scM,
                                    TsExtensionManager* eM,
                                    G4String scorerName, G4String quantity,
                                    G4String outFileName, G4bool isSubScorer);

    ~TsScoreGPUOpticalPhotonFluence() override;

    G4bool ProcessHits(G4Step*, G4TouchableHistory*) override;

    // EndOfEvent 훅: GPU transit hit를 수집하여 빈에 기록
    void UserHookForEndOfEvent() override;

    // Fix C1: Run 종료 훅 — 잔여 배치 flush + 마지막 transit hit 반영
    // (EventBatchSize로 N%BatchSize≠0 인 경우 마지막 이벤트들이 소실되는 문제)
    void UserHookForEndOfRun() override;

    // Fix 70j: Z-fastest CSV 직접 출력 (TOPAS 기본 X-fastest 대체)
    void Output() override;
    void Clear() override;

    // GPU 엔진 참조 설정
    static void SetGPUEngine(TsGPUOpticalPhysics* engine);

private:
    // 컴포넌트 지오메트리 정보 캐시 및 스코어링 볼륨 등록
    void CacheComponentGeometry();

    // DDA 레이 트레이싱: transit hit의 광선을 빈 격자를 통해 추적
    // step_length/volume 누적
    void TraceRayThroughGrid(G4double startX, G4double startY, G4double startZ,
                             G4double dirX, G4double dirY, G4double dirZ,
                             G4double pathLength, G4double photonWeight = 1.0);

    static TsGPUOpticalPhysics* fGPUEngine;

    // 컴포넌트 지오메트리 캐시
    G4double fCompTransX, fCompTransY, fCompTransZ;  // 컴포넌트 월드 위치
    G4double fCompFullX, fCompFullY, fCompFullZ;      // 컴포넌트 전체 크기
    G4int fNBinsX, fNBinsY, fNBinsZ;                  // 빈 수
    // 2026-05-20: energy/wavelength binning (CPU Fluence EBins 동등). fNBinsE=1 = 기존.
    G4int fNBinsE = 1;
    G4double fEBinMinEv = 0.0;
    G4double fEBinMaxEv = 0.0;
    std::vector<double> fGpuFullBins;                  // voxel × nBinsE buffer view (EBins 시)
    G4bool fGeomCached;
    // Phase 2.1+ (2026-05-16): cylindrical/spherical voxelization 지원.
    //   fVoxelType 0=BOX, 1=CYLINDER, 2=SPHERE.
    //   CYL/SPH 인 경우 fRMin/fPhiStart/fThetaStart 가 GPU shader 측 bin 계산에 사용.
    G4int    fVoxelType = 0;
    G4double fRMin = 0;
    G4double fPhiStart = 0;
    G4double fThetaStart = 0;
    G4bool fScoringVolumeRegistered;                   // 스코어링 볼륨 등록 완료 여부
    static G4bool fAABBRegisteredOnce;                 // Fix 49: AABB 등록 + ForceRepropagate 1회만 (이름 기반으로 변경)
    // Multi-scorer race fix (2026-05-01): scorer 이름별 ForceRepropagate dedup.
    // 기존 fAABBRegisteredOnce (전체 1회) → 첫 scorer 만 re-propagate 하고 후속
    // scorer 의 AABB 는 union 으로 추가되지만 engine 에 다시 propagate 안 함 →
    // cached transit hits 가 첫 scorer 의 AABB 만 반영한 stale 상태로 남아
    // 후속 scorer 가 처리 → race + corruption.
    // scorer name 기반으로 각 scorer 첫 등장 시 1회 ForceRepropagate.
    static std::set<std::string> sScorerNamesWithRepropagate;


    // Multi-scorer race fix (2026-05-01): per-scorer accumulating GPU buffer handle.
    // -1 = 미등록. CacheComponentGeometry 첫 호출 시 등록.
    int fScorerBufferHandle = -1;

    // 2026-05-12: harvest callback 등록 여부 — scorer NAME 기준 global.
    // multi-instance (worker thread, ForceRepropagate 등) 시 같은 scorer name 의
    // 2번째 호출은 skip. callback 중복 등록 → DDA 중복 dispatch 방지.
    static std::set<std::string> sScorerNamesWithCallback;

    // Multi-scorer race fix (2026-05-01): scorer 인스턴스 전역 직렬화.
    // 모든 scorer 가 같은 fGPUEngine + 그 안의 engine direct API (DDA dispatch,
    // transit hit cache, DDA bin buffer) 를 공유한다. UserHookForEndOfEvent /
    // UserHookForEndOfRun 이 multi-scorer 동시 실행되면 engine 의 m_pendingDDACmdBuf
    // / DDA bin buffer / transit hit cache 가 race → Apple Metal driver heap
    // corruption (이전 batch=5 SIGSEGV/SIGTRAP 의 root cause). scorer A 가 hook
    // 처리하는 동안 scorer B 는 lock 으로 대기.
    static std::mutex sScorerMutex;

    // 2026-05-02 fix: TOPAS multi-thread 에서 worker 마다 scorer instance 가 따로 만들어지지만
    // GPU 측 buffer 는 scorer NAME 1개당 1개만 등록하여 모든 worker 가 같은 buffer 에 atomic
    // accumulate. Master scorer 만 EndOfRun 에서 그 buffer 를 한 번 read 해서 fFirstMomentMap
    // 으로 transfer.
    static std::map<std::string, int> sScorerNameToHandle;

    // 2026-05-28 multi-scorer fix: 모든 GPU optical scorer 가 harvest 콜백을 등록한 뒤에야
    // propagation 을 1회 트리거해야 한다. 그래야 단일 propagation 의 transit hit 을 harvest
    // 루프가 전 scorer 콜백에 재사용(scorer 별 DDA) → 2번째+ scorer 도 데이터 수신.
    // sAllScorerNames = ctor 에서 모인 전체 GPU scorer 이름(전체 수). 첫 EndOfEvent 시점엔
    // 모든 scorer 가 생성 완료 상태. sScorerNamesWithCallback.size() 와 같아지면 = 마지막 등록.
    static std::set<std::string> sAllScorerNames;
    static thread_local bool sPropagationTriggered;   // 2026-06-08 MT fix: thread_local — worker thread 마다 자기 첫 이벤트 1회 ForceRepropagate. static이면 전역 첫 worker 하나만 트리거되고 나머지 N-1 worker 의 첫 이벤트 genstep 이 소실됨 (loss ≈ (N-1)/Nevents)
};

#endif /* TS_SCORE_GPU_OPTICAL_PHOTON_FLUENCE_HH */
