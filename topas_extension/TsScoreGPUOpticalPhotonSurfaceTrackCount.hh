/**
 * TsScoreGPUOpticalPhotonSurfaceTrackCount.hh
 *
 * SF1 (2026-04-24): GPU surface flux scorer (CPU TsScoreSurfaceTrackCount의 GPU 등가물).
 *
 * 5명 협업 design — bounded plane을 통과하는 광자 카운트.
 *   광학: T+R conservation, Brewster validation 가능
 *   TOPAS: CPU TsScoreSurfaceTrackCount와 1:1 parameter convention
 *   MC: per-crossing count, photonWeight 누적 (RR/splitting ready)
 *   GPU: bounded plane signed-distance 부호 변화 detection
 *   하네스: 옵션 A (thin AABB)와 ±1% cross-check
 *
 * TOPAS 파라미터 사용 예:
 *   s:Sc/MyScorer/Quantity                 = "GPUOpticalPhotonSurfaceTrackCount"
 *   s:Sc/MyScorer/Component                = "Glass"
 *   s:Sc/MyScorer/Surface                  = "Glass/ZPlusSurface"
 *   s:Sc/MyScorer/OnlyIncludeParticlesGoing = "in"   # or "out"
 */

#ifndef TS_SCORE_GPU_OPTICAL_PHOTON_SURFACE_TRACK_COUNT_HH
#define TS_SCORE_GPU_OPTICAL_PHOTON_SURFACE_TRACK_COUNT_HH

#include "TsVBinnedScorer.hh"
#include "MOPTypes.hh"  // wrapper → ../include/MOPTypes.h (canonical)

#include <vector>
#include <cstdint>

class TsGPUOpticalPhysics;

class TsScoreGPUOpticalPhotonSurfaceTrackCount : public TsVBinnedScorer {
public:
    TsScoreGPUOpticalPhotonSurfaceTrackCount(TsParameterManager* pM, TsMaterialManager* mM,
                                              TsGeometryManager* gM, TsScoringManager* scM,
                                              TsExtensionManager* eM,
                                              G4String scorerName, G4String quantity,
                                              G4String outFileName, G4bool isSubScorer);
    ~TsScoreGPUOpticalPhotonSurfaceTrackCount() override;

    G4bool ProcessHits(G4Step* aStep, G4TouchableHistory*) override;
    void UpdateForNewRun(G4bool rebuiltSomeComponents) override;  // SF1 deficit fix: pre-register surface
    void UserHookForEndOfEvent() override;
    void UserHookForEndOfRun() override;
    void Output() override;
    void Clear() override;

    static void SetGPUEngine(TsGPUOpticalPhysics* engine);

private:
    static TsGPUOpticalPhysics* fGPUEngine;

    // Surface registration
    bool fSurfaceRegistered;
    uint32_t fSurfaceId;
    static uint32_t fNextSurfaceId;  // scorer 등록 순번 (0-based, 개수 제한 없음)

    // Filter direction (parsed from OnlyIncludeParticlesGoing)
    bool fFilterIn;     // count only "going in" (cosθ < 0)
    bool fFilterOut;    // count only "going out" (cosθ ≥ 0)

    // Counter (single bin: total surface crossings)
    G4double fAccumulatedCount;

    // Bug A fix: cache version 추적 — 같은 batch 중복 처리 방지
    uint64_t fLastProcessedCacheVersion;

    void RegisterSurfaceFromTOPASParam();
};

#endif /* TS_SCORE_GPU_OPTICAL_PHOTON_SURFACE_TRACK_COUNT_HH */
