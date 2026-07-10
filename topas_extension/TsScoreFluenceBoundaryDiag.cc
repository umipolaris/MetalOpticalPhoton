// Scorer for FluenceBoundaryDiag
#include "TsScoreFluenceBoundaryDiag.hh"

#include "G4OpBoundaryProcess.hh"
#include "G4ProcessManager.hh"
#include "G4OpticalPhoton.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4VPhysicalVolume.hh"
#include "G4LogicalBorderSurface.hh"
#include "G4LogicalSkinSurface.hh"
#include "G4SystemOfUnits.hh"

#include <iostream>

TsScoreFluenceBoundaryDiag::TsScoreFluenceBoundaryDiag(
    TsParameterManager* pM, TsMaterialManager* mM, TsGeometryManager* gM,
    TsScoringManager* scM, TsExtensionManager* eM,
    G4String scorerName, G4String quantity, G4String outFileName, G4bool isSubScorer)
    : TsVBinnedScorer(pM, mM, gM, scM, eM, scorerName, quantity, outFileName, isSubScorer)
    , fBoundaryProcess(nullptr)
    , fBoundaryProcessFound(false)
    , fSurfTotal(0), fSurfReflect(0), fSurfTransmit(0)
    , fNoSurfTotal(0), fNoSurfReflect(0), fNoSurfTransmit(0)
    , fFatePhotons(0), fFateBulkAbs(0)
    , fFateSideEscXm(0), fFateSideEscXp(0), fFateSideEscYm(0), fFateSideEscYp(0)
    , fFateEndEscZm(0), fFateEndEscZp(0)
    , fFateBoundaryAbs(0), fFateOther(0)
    , fEmitDirCount(0)
{
    SetUnit("/mm2");
    for (int i = 0; i < 8; i++) fReflHist[i] = 0;
    for (int i = 0; i < 20; i++) fEmitCosZHist[i] = 0;
    for (int i = 0; i < 18; i++) fEmitPhiHist[i] = 0;
}

TsScoreFluenceBoundaryDiag::~TsScoreFluenceBoundaryDiag() {}

namespace {
void ClassifyAndCountFate(G4Step* aStep,
    uint64_t& fFatePhotons, uint64_t& fFateBulkAbs,
    uint64_t& fFateSideEscXm, uint64_t& fFateSideEscXp,
    uint64_t& fFateSideEscYm, uint64_t& fFateSideEscYp,
    uint64_t& fFateEndEscZm, uint64_t& fFateEndEscZp,
    uint64_t& fFateBoundaryAbs, uint64_t& fFateOther,
    uint64_t fReflHist[8],
    std::map<G4int, uint32_t>& fTrackBoundaryAttempts);
}

G4OpBoundaryProcess* TsScoreFluenceBoundaryDiag::FindBoundaryProcess()
{
    G4ProcessManager* pm = G4OpticalPhoton::OpticalPhoton()->GetProcessManager();
    if (!pm) return nullptr;

    G4int nProc = pm->GetProcessListLength();
    G4ProcessVector* pv = pm->GetProcessList();
    for (G4int i = 0; i < nProc; i++) {
        G4OpBoundaryProcess* bp = dynamic_cast<G4OpBoundaryProcess*>((*pv)[i]);
        if (bp) return bp;
    }
    return nullptr;
}

G4bool TsScoreFluenceBoundaryDiag::ProcessHits(G4Step* aStep, G4TouchableHistory*)
{
    // [EmitDirDiag-DIRECT 2026-04-28] print all opticalphoton ProcessHits with step number
    // [주석처리 2026-05-14] guard 없는 매-event 스팸 출력 — 디버그용
    // {
    //     G4Track* trk = aStep->GetTrack();
    //     if (trk->GetDefinition() == G4OpticalPhoton::OpticalPhoton()) {
    //         G4ThreeVector dir = aStep->GetPreStepPoint()->GetMomentumDirection();
    //         std::cout << "[EMITDIR-CPU] step=" << trk->GetCurrentStepNumber()
    //                   << " trk=" << trk->GetTrackID()
    //                   << " cosθ=" << dir.z()
    //                   << " phi=" << std::atan2(dir.y(), dir.x())
    //                   << std::endl;
    //     }
    // }
    if (!fIsActive) {
        fSkippedWhileInactive++;
        return false;
    }

    // 표준 Fluence 스코어링
    G4double quantity = aStep->GetStepLength();
    if (quantity > 0.) {
        ResolveSolid(aStep);
        quantity /= GetCubicVolume(aStep);
        quantity *= aStep->GetPreStepPoint()->GetWeight();
        AccumulateHit(aStep, quantity);
    }

    // [EmitDirDiag 2026-04-28] 첫 step (track 생성 직후) emit direction 캡처
    // GetCurrentStepNumber() == 1 이 첫 step (G4 1-based). step 0이면 아직 transport 전.
    // 하지만 ProcessHits 는 transport 후 호출되므로 step==1 = first step
    {
        G4Track* trk = aStep->GetTrack();
        if (trk->GetDefinition() == G4OpticalPhoton::OpticalPhoton()) {
            G4int trackId = trk->GetTrackID();
            if (fTrackSeenEmit.find(trackId) == fTrackSeenEmit.end()) {
                fTrackSeenEmit[trackId] = true;
                // pre-step direction: 첫 step의 direction은 emit direction (boundary 미처리 전)
                G4ThreeVector preDir = aStep->GetPreStepPoint()->GetMomentumDirection();
                double dz = preDir.z();
                double dx = preDir.x();
                double dy = preDir.y();
                int cz_bin = (int)((dz + 1.0) * 10.0);
                if (cz_bin < 0) cz_bin = 0;
                if (cz_bin >= 20) cz_bin = 19;
                fEmitCosZHist[cz_bin]++;
                double phi_emit = std::atan2(dy, dx);
                int phi_bin = (int)((phi_emit + 3.14159265) / 6.28318530 * 18.0);
                if (phi_bin < 0) phi_bin = 0;
                if (phi_bin >= 18) phi_bin = 17;
                fEmitPhiHist[phi_bin]++;
                fEmitDirCount++;
            }
        }
    }

    // [FateMatrix-CPU 2026-04-27] photon track 종료 시점 fate 분류 (boundary 처리와 무관)
    ClassifyAndCountFate(aStep,
        fFatePhotons, fFateBulkAbs,
        fFateSideEscXm, fFateSideEscXp, fFateSideEscYm, fFateSideEscYp,
        fFateEndEscZm, fFateEndEscZp,
        fFateBoundaryAbs, fFateOther,
        fReflHist, fTrackBoundaryAttempts);

    // 경계면 진단: optical photon만
    fNoSurfTotal++;  // DEBUG: ProcessHits 호출 횟수
    if (aStep->GetTrack()->GetDefinition() != G4OpticalPhoton::OpticalPhoton())
        return (quantity > 0.);

    // G4OpBoundaryProcess 찾기 (최초 1회)
    if (!fBoundaryProcessFound) {
        fBoundaryProcess = FindBoundaryProcess();
        fBoundaryProcessFound = true;
        if (fBoundaryProcess) {
            std::cout << "[BndDiag] G4OpBoundaryProcess found" << std::endl;
        } else {
            std::cout << "[BndDiag] WARNING: G4OpBoundaryProcess NOT found" << std::endl;
        }
    }

    if (!fBoundaryProcess) return (quantity > 0.);

    G4OpBoundaryProcessStatus status = fBoundaryProcess->GetStatus();

    // 경계 이벤트가 아니면 스킵 (Undefined = 경계 아님)
    if (status == Undefined || status == NotAtBoundary || status == SameMaterial ||
        status == StepTooSmall || status == NoRINDEX)
        return (quantity > 0.);

    // 표면 유무 확인: pre/post 볼륨 사이에 BorderSurface 또는 SkinSurface가 있는지
    G4StepPoint* postPt = aStep->GetPostStepPoint();
    G4VPhysicalVolume* preVol = aStep->GetPreStepPoint()->GetPhysicalVolume();
    G4VPhysicalVolume* postVol = postPt->GetPhysicalVolume();

    G4bool hasSurface = false;
    if (preVol && postVol) {
        G4LogicalSurface* surf = G4LogicalBorderSurface::GetSurface(preVol, postVol);
        if (!surf) surf = G4LogicalBorderSurface::GetSurface(postVol, preVol);
        if (!surf) surf = G4LogicalSkinSurface::GetSurface(preVol->GetLogicalVolume());
        if (!surf) surf = G4LogicalSkinSurface::GetSurface(postVol->GetLogicalVolume());
        hasSurface = (surf != nullptr);
    }

    // 투과 vs 반사 판별
    G4bool transmitted = false;
    switch (status) {
        case FresnelRefraction:
        case Transmission:
        case CoatedDielectricRefraction:
        case CoatedDielectricFrustratedTransmission:
            transmitted = true;
            break;
        case FresnelReflection:
        case TotalInternalReflection:
        case LambertianReflection:
        case LobeReflection:
        case SpikeReflection:
        case BackScattering:
        case CoatedDielectricReflection:
            transmitted = false;
            break;
        default:
            // Absorption, Detection, etc. — 경계 반사/투과가 아님, 스킵
            return (quantity > 0.);
    }

    if (hasSurface) {
        fSurfTotal++;
        if (transmitted) fSurfTransmit++;
        else fSurfReflect++;
    } else {
        fNoSurfTotal++;
        if (transmitted) fNoSurfTransmit++;
        else fNoSurfReflect++;
    }

    // [FateMatrix-CPU 2026-04-27] Boundary attempt 카운트 (per-photon track)
    // status 가 boundary 처리 일어남 (transmitted or reflected) 일 때 +1
    G4int trackId = aStep->GetTrack()->GetTrackID();
    fTrackBoundaryAttempts[trackId]++;

    // [WB-CPU-PUSH 2026-04-28] Removed (G4 Navigator snap-back).
    // Production W-B push moved to G4OpBoundaryProcess.cc (Geant4 source patch).

    return (quantity > 0.);
}

namespace {
// Fate matrix 분류 helper
void ClassifyAndCountFate(G4Step* aStep,
    uint64_t& fFatePhotons, uint64_t& fFateBulkAbs,
    uint64_t& fFateSideEscXm, uint64_t& fFateSideEscXp,
    uint64_t& fFateSideEscYm, uint64_t& fFateSideEscYp,
    uint64_t& fFateEndEscZm, uint64_t& fFateEndEscZp,
    uint64_t& fFateBoundaryAbs, uint64_t& fFateOther,
    uint64_t fReflHist[8],
    std::map<G4int, uint32_t>& fTrackBoundaryAttempts)
{
    G4Track* track = aStep->GetTrack();
    if (track->GetTrackStatus() != fStopAndKill) return;
    if (track->GetDefinition() != G4OpticalPhoton::OpticalPhoton()) return;

    G4int trackId = track->GetTrackID();
    fFatePhotons++;

    // post-step process name 으로 fate 결정
    G4StepPoint* postPt = aStep->GetPostStepPoint();
    const G4VProcess* postProc = postPt->GetProcessDefinedStep();
    G4String pname = postProc ? postProc->GetProcessName() : "";

    if (pname == "OpAbsorption") {
        fFateBulkAbs++;
    } else if (postPt->GetPhysicalVolume() == nullptr ||
               pname == "Transportation") {
        // World boundary escape — face 분류 by post position
        G4ThreeVector pos = postPt->GetPosition() / CLHEP::mm;
        G4double ax = std::fabs(pos.x()), ay = std::fabs(pos.y()), az = std::fabs(pos.z());
        // BC408 = ±25 (X,Y) × ±150 (Z)
        G4double dx = ax - 25.0, dy = ay - 25.0, dz = az - 150.0;
        if (dx >= dy && dx >= dz) {
            if (pos.x() < 0) fFateSideEscXm++; else fFateSideEscXp++;
        } else if (dy >= dz) {
            if (pos.y() < 0) fFateSideEscYm++; else fFateSideEscYp++;
        } else {
            if (pos.z() < 0) fFateEndEscZm++; else fFateEndEscZp++;
        }
    } else if (pname == "OpBoundary") {
        fFateBoundaryAbs++;
    } else {
        fFateOther++;
    }

    // reflectedCount histogram
    auto it = fTrackBoundaryAttempts.find(trackId);
    uint32_t attempts = (it != fTrackBoundaryAttempts.end()) ? it->second : 0u;
    int rb = 0;
    if      (attempts == 0u) rb = 0;
    else if (attempts == 1u) rb = 1;
    else if (attempts == 2u) rb = 2;
    else if (attempts == 3u) rb = 3;
    else if (attempts == 4u) rb = 4;
    else if (attempts < 10u) rb = 5;
    else if (attempts < 100u) rb = 6;
    else                       rb = 7;
    fReflHist[rb]++;

    // map cleanup (track 종료, no longer needed)
    if (it != fTrackBoundaryAttempts.end()) fTrackBoundaryAttempts.erase(it);
}
}  // namespace

void TsScoreFluenceBoundaryDiag::UserHookForEndOfRun()
{
    // [EmitDirDiag 2026-04-28] CPU emit direction histogram
    std::cout << "[EmitDirDiag-CPU] Total emit photons: " << fEmitDirCount
              << "  (ProcessHits 호출 카운트 = fNoSurfTotal=" << fNoSurfTotal
              << ", track 맵 size=" << fTrackSeenEmit.size() << ")" << std::endl;
    std::cout << "[EmitDirDiag-CPU] cosθ histogram (20 bins ∈ [-1,+1], isotropic = 5%/bin):" << std::endl;
    for (int b = 0; b < 20; b++) {
        double lo = -1.0 + b * 0.1;
        double pct = (fEmitDirCount > 0) ? 100.0 * fEmitCosZHist[b] / fEmitDirCount : 0;
        std::cout << "[EmitDirDiag-CPU]   cosθ=[" << lo << "," << lo+0.1 << "]: "
                  << fEmitCosZHist[b] << " (" << pct << "%)" << std::endl;
    }
    std::cout << "[EmitDirDiag-CPU] φ histogram (18 bins ∈ [-π,+π], isotropic = 5.56%/bin):" << std::endl;
    for (int b = 0; b < 18; b++) {
        double pct = (fEmitDirCount > 0) ? 100.0 * fEmitPhiHist[b] / fEmitDirCount : 0;
        std::cout << "[EmitDirDiag-CPU]   φ_bin" << b << ": "
                  << fEmitPhiHist[b] << " (" << pct << "%)" << std::endl;
    }

    std::cout << "[BndDiag] === CPU Boundary Diagnostics ===" << std::endl;
    std::cout << "[BndDiag]   WithSurface: total=" << fSurfTotal
              << " reflect=" << fSurfReflect
              << " transmit=" << fSurfTransmit;
    if (fSurfTotal > 0)
        std::cout << " (T%=" << (100.0 * fSurfTransmit / fSurfTotal) << "%)";
    std::cout << std::endl;
    std::cout << "[BndDiag]   NoSurface:   total=" << fNoSurfTotal
              << " reflect=" << fNoSurfReflect
              << " transmit=" << fNoSurfTransmit;
    if (fNoSurfTotal > 0)
        std::cout << " (T%=" << (100.0 * fNoSurfTransmit / fNoSurfTotal) << "%)";
    std::cout << std::endl;

    // [FateMatrix-CPU 2026-04-27] Per-photon final fate 출력
    // Metric 정의: G4Track 의 fStopAndKill 시점에 photon 한 번만 카운트.
    // boundary attempts (multi-bounce) 와는 별개 — photon lifetime 종료 시 fate.
    std::cout << "[FateMatrix-CPU] === Per-photon final fate (unique end-of-track) ===" << std::endl;
    std::cout << "[FateMatrix-CPU]   total photons classified: " << fFatePhotons << std::endl;
    std::cout << "[FateMatrix-CPU]   bulk_abs (OpAbsorption):  " << fFateBulkAbs << std::endl;
    std::cout << "[FateMatrix-CPU]   side_escape: -X=" << fFateSideEscXm
              << " +X=" << fFateSideEscXp
              << " -Y=" << fFateSideEscYm
              << " +Y=" << fFateSideEscYp << std::endl;
    std::cout << "[FateMatrix-CPU]   end_escape:  -Z=" << fFateEndEscZm
              << " +Z=" << fFateEndEscZp << std::endl;
    std::cout << "[FateMatrix-CPU]   side(X+Y) sum=" << (fFateSideEscXm+fFateSideEscXp+fFateSideEscYm+fFateSideEscYp)
              << "  end(-Z+Z) sum=" << (fFateEndEscZm+fFateEndEscZp) << std::endl;
    std::cout << "[FateMatrix-CPU]   boundary_abs (OpBoundary): " << fFateBoundaryAbs << std::endl;
    std::cout << "[FateMatrix-CPU]   other: " << fFateOther << std::endl;
    std::cout << "[FateMatrix-CPU] reflectedCount histogram (per-photon boundary attempts):"
              << " r=0:" << fReflHist[0]
              << " r=1:" << fReflHist[1]
              << " r=2:" << fReflHist[2]
              << " r=3:" << fReflHist[3]
              << " r=4:" << fReflHist[4]
              << " r=5-9:" << fReflHist[5]
              << " r=10-99:" << fReflHist[6]
              << " r>=100:" << fReflHist[7] << std::endl;
}
