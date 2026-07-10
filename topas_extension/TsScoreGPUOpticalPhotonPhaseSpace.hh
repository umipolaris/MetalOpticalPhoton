/**
 * TsScoreGPUOpticalPhotonPhaseSpace.hh
 * TOPAS Extension Scorer: GPU 광학 광자 Phase-Space Dump
 *
 * 2026-05-19: 광자 별 phase-space record (transit hit dump) → CSV.
 * MOPEngine_GetTransitHits() → host MOPHit array → per-photon ASCII export.
 *
 * MOPHit fields: posX, posY, posZ, time, energy (eV), wavelength (nm),
 *                volumeId, parentTrackId, flags
 *
 * TOPAS 사용:
 *   s:Sc/MyPhSp/Quantity = "GPUOpticalPhotonPhaseSpace"
 *   s:Sc/MyPhSp/Component = "ScorePlane"
 *   s:Sc/MyPhSp/OutputType = "csv"
 *   s:Sc/MyPhSp/OutputFile = "photon_phsp"
 */

#ifndef TS_SCORE_GPU_OPTICAL_PHOTON_PHASE_SPACE_HH
#define TS_SCORE_GPU_OPTICAL_PHOTON_PHASE_SPACE_HH

#include "TsVNtupleScorer.hh"
#include <vector>
#include <string>

class TsGPUOpticalPhysics;
// MOPHit 은 typedef (MOPTypes.h) — forward declare 생략, .cc 에서 #include "MOPTypes.h"

class TsScoreGPUOpticalPhotonPhaseSpace : public TsVNtupleScorer {
public:
    TsScoreGPUOpticalPhotonPhaseSpace(TsParameterManager* pM, TsMaterialManager* mM,
                                       TsGeometryManager* gM, TsScoringManager* scM,
                                       TsExtensionManager* eM,
                                       G4String scorerName, G4String quantity,
                                       G4String outFileName, G4bool isSubScorer);

    ~TsScoreGPUOpticalPhotonPhaseSpace() override;

    G4bool ProcessHits(G4Step*, G4TouchableHistory*) override;
    void UpdateForNewRun(G4bool rebuiltSomeComponents) override;
    void UserHookForEndOfEvent() override;
    void UserHookForEndOfRun() override;

    static void SetGPUEngine(TsGPUOpticalPhysics* engine);

private:
    static TsGPUOpticalPhysics* fGPUEngine;
    std::string fCsvPath;
    uint64_t fTotalRecords;
};

#endif
