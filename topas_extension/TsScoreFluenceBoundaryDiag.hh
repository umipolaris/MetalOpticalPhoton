#ifndef TsScoreFluenceBoundaryDiag_hh
#define TsScoreFluenceBoundaryDiag_hh

#include "TsVBinnedScorer.hh"
#include <map>

class G4OpBoundaryProcess;

class TsScoreFluenceBoundaryDiag : public TsVBinnedScorer
{
public:
    TsScoreFluenceBoundaryDiag(TsParameterManager* pM, TsMaterialManager* mM,
        TsGeometryManager* gM, TsScoringManager* scM, TsExtensionManager* eM,
        G4String scorerName, G4String quantity, G4String outFileName, G4bool isSubScorer);

    virtual ~TsScoreFluenceBoundaryDiag();

    G4bool ProcessHits(G4Step*, G4TouchableHistory*);
    void UserHookForEndOfRun();

private:
    G4OpBoundaryProcess* FindBoundaryProcess();
    G4OpBoundaryProcess* fBoundaryProcess;
    G4bool fBoundaryProcessFound;

    // 경계면 진단 카운터
    uint64_t fSurfTotal;
    uint64_t fSurfReflect;
    uint64_t fSurfTransmit;
    uint64_t fNoSurfTotal;
    uint64_t fNoSurfReflect;
    uint64_t fNoSurfTransmit;

    // [FateMatrix-CPU 2026-04-27] Per-photon final fate (unique photon end-of-track).
    // Metric 정의: 한 photon 의 G4Track 이 fStopAndKill 될 때 한 번만 카운트.
    // boundary attempts (multi-bounce) 와는 별개 — 각 photon 의 lifetime 종료 시 fate.
    //
    // Fate 분류 (mutually exclusive):
    //   bulk_abs: postStep process == OpAbsorption (BC408 안 absorb)
    //   side_esc_xm/xp/ym/yp: World boundary -X/+X/-Y/+Y face escape (post volume == nullptr)
    //   end_esc_zm/zp: World boundary -Z/+Z face escape
    //   boundary_abs: post process == OpBoundary, status == Absorption (surface absorber)
    //   other: 분류 안 됨
    uint64_t fFatePhotons;       // 분류된 photon 총 개수
    uint64_t fFateBulkAbs;
    uint64_t fFateSideEscXm, fFateSideEscXp, fFateSideEscYm, fFateSideEscYp;
    uint64_t fFateEndEscZm, fFateEndEscZp;
    uint64_t fFateBoundaryAbs;
    uint64_t fFateOther;

    // Per-photon reflectedCount (boundary attempts) histogram, max 31.
    // Track 별 boundary attempts 카운트 → fStopAndKill 시 histogram bin 증가.
    // bins: 0, 1, 2, 3, 4, 5-9, 10-99, 100+
    uint64_t fReflHist[8];
    std::map<G4int, uint32_t> fTrackBoundaryAttempts;  // trackID → attempts count

    // [EmitDirDiag 2026-04-28] emit direction histogram per opticalphoton track
    // step number == 0 시 capture (direction is emit direction before any boundary processing)
    uint64_t fEmitCosZHist[20];     // cosθ ∈ [-1, +1], 20 bins of 0.1
    uint64_t fEmitPhiHist[18];      // φ ∈ [-π, +π], 18 bins of 20°
    uint64_t fEmitDirCount;
    std::map<G4int, bool> fTrackSeenEmit;  // already captured emit dir for this track
};

#endif
