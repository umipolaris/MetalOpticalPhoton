/**
 * TsScoreGPUOpticalPhotonCount.hh
 * TOPAS Extension Scorer: GPU 광학 광자 검출 스코어러
 *
 * GPU에서 전파된 광학 광자의 히트를 수집하여 CSV/ROOT로 출력
 *
 * TOPAS 파라미터 파일에서 사용:
 *   s:Sc/MyScorer/Quantity = "GPUOpticalPhotonCount"
 *   s:Sc/MyScorer/Component = "DetectorVolume"
 *   s:Sc/MyScorer/OutputType = "CSV"
 *   s:Sc/MyScorer/OutputFile = "OpticalPhotonHits"
 */

#ifndef TS_SCORE_GPU_OPTICAL_PHOTON_COUNT_HH
#define TS_SCORE_GPU_OPTICAL_PHOTON_COUNT_HH

#include "TsVNtupleScorer.hh"

class TsGPUOpticalPhysics;

class TsScoreGPUOpticalPhotonCount : public TsVNtupleScorer {
public:
    TsScoreGPUOpticalPhotonCount(TsParameterManager* pM, TsMaterialManager* mM,
                                  TsGeometryManager* gM, TsScoringManager* scM,
                                  TsExtensionManager* eM,
                                  G4String scorerName, G4String quantity,
                                  G4String outFileName, G4bool isSubScorer);

    ~TsScoreGPUOpticalPhotonCount() override;

    G4bool ProcessHits(G4Step*, G4TouchableHistory*) override;

    // EndOfEvent 훅: GPU 히트를 수집하여 Ntuple에 기록
    void UserHookForEndOfEvent() override;

    // Fix C1: Run 종료 훅 — 잔여 배치 flush + 마지막 hit 기록
    void UserHookForEndOfRun() override;

    // GPU 엔진 참조 설정
    static void SetGPUEngine(TsGPUOpticalPhysics* engine);

protected:
    G4float fPosX;
    G4float fPosY;
    G4float fPosZ;
    G4float fWaveLength;
    G4float fTime;
    G4float fEnergy;
    G4int   fVolumeId;
    G4int   fParentTrackId;

private:
    static TsGPUOpticalPhysics* fGPUEngine;
};

#endif /* TS_SCORE_GPU_OPTICAL_PHOTON_COUNT_HH */
