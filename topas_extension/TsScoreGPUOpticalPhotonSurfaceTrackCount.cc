// Scorer for GPUOpticalPhotonSurfaceTrackCount
//
// SF1 (2026-04-24): GPU surface flux scorer 구현 (5명 협업 design).

#include "TsScoreGPUOpticalPhotonSurfaceTrackCount.hh"
#include "TsGPUOpticalPhysics.hh"
#include "TsParameterManager.hh"
#include "TsVGeometryComponent.hh"
#include "G4SystemOfUnits.hh"

#include <iostream>
#include <cstring>
#include <algorithm>
#include <set>
#include <tuple>
#include <map>

TsGPUOpticalPhysics* TsScoreGPUOpticalPhotonSurfaceTrackCount::fGPUEngine = nullptr;
uint32_t TsScoreGPUOpticalPhotonSurfaceTrackCount::fNextSurfaceId = 0;

// Dedup map: same (Component, Surface) key → reuse same surfaceId (avoid TOPAS subscorer double-registration).
static std::map<std::string, uint32_t> s_registeredSurfaces;

TsScoreGPUOpticalPhotonSurfaceTrackCount::TsScoreGPUOpticalPhotonSurfaceTrackCount(
    TsParameterManager* pM, TsMaterialManager* mM, TsGeometryManager* gM,
    TsScoringManager* scM, TsExtensionManager* eM,
    G4String scorerName, G4String quantity,
    G4String outFileName, G4bool isSubScorer)
    : TsVBinnedScorer(pM, mM, gM, scM, eM, scorerName, quantity, outFileName, isSubScorer)
    , fSurfaceRegistered(false)
    , fSurfaceId(0xFFFFFFFFu)
    , fFilterIn(false)
    , fFilterOut(false)
    , fAccumulatedCount(0.0)
    , fLastProcessedCacheVersion(0)
{
    SetSurfaceScorer();   // 5명 합의: surface scorer로 등록 (CPU TsScoreSurfaceTrackCount와 동일)
    SetUnit("");

    // OnlyIncludeParticlesGoing 파라미터 파싱 (CPU TsScoreSurfaceTrackCount convention)
    G4String going_param = GetFullParmName("OnlyIncludeParticlesGoing");
    if (fPm->ParameterExists(going_param)) {
        G4String going = fPm->GetStringParameter(going_param);
        if (going == "In" || going == "in")  fFilterIn = true;
        if (going == "Out" || going == "out") fFilterOut = true;
    }
}

TsScoreGPUOpticalPhotonSurfaceTrackCount::~TsScoreGPUOpticalPhotonSurfaceTrackCount() {
    std::cout << "[TsGPU-SF1] FINAL: scorer '" << GetName() << "' surfaceId=" << fSurfaceId
              << " accumulated count=" << fAccumulatedCount
              << " filter=" << (fFilterIn ? "in" : (fFilterOut ? "out" : "both"))
              << std::endl;
}

void TsScoreGPUOpticalPhotonSurfaceTrackCount::SetGPUEngine(TsGPUOpticalPhysics* engine) {
    fGPUEngine = engine;
}

void TsScoreGPUOpticalPhotonSurfaceTrackCount::RegisterSurfaceFromTOPASParam() {
    if (fSurfaceRegistered || !fGPUEngine || !fComponent) return;

    // Surface 파라미터 파싱: "Component/FaceName" (CPU와 동일 convention)
    G4String surfaceParm = GetFullParmName("Surface");
    G4String faceName = "ZPlusSurface";  // default
    if (fPm->ParameterExists(surfaceParm)) {
        G4String volPlusSurf = fPm->GetStringParameter(surfaceParm);
        size_t pos = volPlusSurf.find_last_of("/");
        if (pos != std::string::npos) {
            faceName = volPlusSurf.substr(pos + 1);
        }
    }

    // Dedup: 같은 (component, faceName) 이 이미 등록되어 있으면 재사용 (subscorer 중복 차단).
    std::string dedupKey = std::string(fComponentName) + "/" + std::string(faceName);
    auto itDedup = s_registeredSurfaces.find(dedupKey);
    if (itDedup != s_registeredSurfaces.end()) {
        fSurfaceId = itDedup->second;
        fSurfaceRegistered = true;
        std::cout << "[TsGPU-SF1] Reusing existing surface '" << GetName()
                  << "' face=" << faceName
                  << " surfaceId=" << fSurfaceId
                  << " (dedup hit for " << dedupKey << ")"
                  << std::endl;
        return;
    }

    // Component AABB로부터 face origin/normal/extents 계산
    G4double fullX = fComponent->GetFullWidth(0);
    G4double fullY = fComponent->GetFullWidth(1);
    G4double fullZ = fComponent->GetFullWidth(2);
    G4double halfX = fullX * 0.5;
    G4double halfY = fullY * 0.5;
    G4double halfZ = fullZ * 0.5;

    // Component world position
    G4double tx = 0, ty = 0, tz = 0;
    {
        G4String compName = fComponentName;
        size_t lastUnderscore = compName.rfind('_');
        if (lastUnderscore != std::string::npos) {
            G4String suffix = compName.substr(lastUnderscore + 1);
            if (suffix.find('x') != std::string::npos) {
                G4String orig = compName.substr(0, lastUnderscore);
                if (fPm->ParameterExists("Ge/" + orig + "/Parent")) compName = orig;
            }
        }
        while (true) {
            G4String prefix = "Ge/" + compName + "/";
            if (fPm->ParameterExists(prefix + "TransX"))
                tx += fPm->GetDoubleParameter(prefix + "TransX", "Length");
            if (fPm->ParameterExists(prefix + "TransY"))
                ty += fPm->GetDoubleParameter(prefix + "TransY", "Length");
            if (fPm->ParameterExists(prefix + "TransZ"))
                tz += fPm->GetDoubleParameter(prefix + "TransZ", "Length");
            G4String parentParm = prefix + "Parent";
            if (!fPm->ParameterExists(parentParm)) break;
            compName = fPm->GetStringParameter(parentParm);
            if (compName == "World" || compName == "world") break;
        }
    }

    MOPSurfaceDef def;
    memset(&def, 0, sizeof(def));
    def.surfaceId = fNextSurfaceId++;
    def.enabled = 1;
    def.originX = (float)(tx / CLHEP::mm);
    def.originY = (float)(ty / CLHEP::mm);
    def.originZ = (float)(tz / CLHEP::mm);

    if (faceName == "ZPlusSurface" || faceName == "ZMinusSurface") {
        float sign = (faceName == "ZPlusSurface") ? +1.0f : -1.0f;
        def.normalX = 0; def.normalY = 0; def.normalZ = sign;
        def.axisUX = 1; def.axisUY = 0; def.axisUZ = 0;
        def.axisVX = 0; def.axisVY = 1; def.axisVZ = 0;
        def.halfExtentU = (float)(halfX / CLHEP::mm);
        def.halfExtentV = (float)(halfY / CLHEP::mm);
        def.originZ += sign * (float)(halfZ / CLHEP::mm);
    } else if (faceName == "XPlusSurface" || faceName == "XMinusSurface") {
        float sign = (faceName == "XPlusSurface") ? +1.0f : -1.0f;
        def.normalX = sign; def.normalY = 0; def.normalZ = 0;
        def.axisUX = 0; def.axisUY = 1; def.axisUZ = 0;
        def.axisVX = 0; def.axisVY = 0; def.axisVZ = 1;
        def.halfExtentU = (float)(halfY / CLHEP::mm);
        def.halfExtentV = (float)(halfZ / CLHEP::mm);
        def.originX += sign * (float)(halfX / CLHEP::mm);
    } else if (faceName == "YPlusSurface" || faceName == "YMinusSurface") {
        float sign = (faceName == "YPlusSurface") ? +1.0f : -1.0f;
        def.normalX = 0; def.normalY = sign; def.normalZ = 0;
        def.axisUX = 1; def.axisUY = 0; def.axisUZ = 0;
        def.axisVX = 0; def.axisVY = 0; def.axisVZ = 1;
        def.halfExtentU = (float)(halfX / CLHEP::mm);
        def.halfExtentV = (float)(halfZ / CLHEP::mm);
        def.originY += sign * (float)(halfY / CLHEP::mm);
    } else {
        std::cerr << "[TsGPU-SF1] Unknown face: " << faceName
                  << " — defaulting to ZPlusSurface" << std::endl;
        def.normalX = 0; def.normalY = 0; def.normalZ = 1;
        def.axisUX = 1; def.axisUY = 0; def.axisUZ = 0;
        def.axisVX = 0; def.axisVY = 1; def.axisVZ = 0;
        def.halfExtentU = (float)(halfX / CLHEP::mm);
        def.halfExtentV = (float)(halfY / CLHEP::mm);
        def.originZ += (float)(halfZ / CLHEP::mm);
    }

    fSurfaceId = fGPUEngine->RegisterSurfaceScorer(def);
    fSurfaceRegistered = (fSurfaceId != 0xFFFFFFFFu);
    if (fSurfaceRegistered) {
        s_registeredSurfaces[dedupKey] = fSurfaceId;
    }

    std::cout << "[TsGPU-SF1] Registered surface '" << GetName()
              << "' face=" << faceName
              << " surfaceId=" << fSurfaceId
              << " origin=(" << def.originX << "," << def.originY << "," << def.originZ << ")"
              << " normal=(" << def.normalX << "," << def.normalY << "," << def.normalZ << ")"
              << " halfExt=(" << def.halfExtentU << "," << def.halfExtentV << ")"
              << " filter=" << (fFilterIn ? "in" : (fFilterOut ? "out" : "both"))
              << std::endl;
}

G4bool TsScoreGPUOpticalPhotonSurfaceTrackCount::ProcessHits(G4Step*, G4TouchableHistory*) {
    return false;  // GPU 광자는 G4Step 안 옴
}

void TsScoreGPUOpticalPhotonSurfaceTrackCount::UpdateForNewRun(G4bool rebuiltSomeComponents) {
    TsVBinnedScorer::UpdateForNewRun(rebuiltSomeComponents);
    // SF1 deficit fix: BeginOfRun에서 surface 등록.
    // Accumulator reset은 첫 scorer 한 번만 (static flag). 다중 BeginOfRun race 회피.
    if (fGPUEngine) {
        if (!fSurfaceRegistered) RegisterSurfaceFromTOPASParam();
        static bool s_accReset = false;
        if (!s_accReset) {
            fGPUEngine->ResetAccumulatedSurfaceHits();
            s_accReset = true;
        }
    }
    fLastProcessedCacheVersion = 0;
    fAccumulatedCount = 0.0;
}

void TsScoreGPUOpticalPhotonSurfaceTrackCount::UserHookForEndOfEvent() {
    if (!fGPUEngine) return;
    if (!fSurfaceRegistered) RegisterSurfaceFromTOPASParam();  // fallback

    fGPUEngine->EndOfEvent();

    // cacheVer-based dedup: same batch을 두 번 처리하지 않음
    uint64_t currentVersion = fGPUEngine->GetCacheVersion();
    if (currentVersion <= fLastProcessedCacheVersion) return;
    fLastProcessedCacheVersion = currentVersion;

    const auto& surfaceHits = fGPUEngine->GetCachedSurfaceHits();
    for (const auto& sh : surfaceHits) {
        if (sh.surfaceId != fSurfaceId) continue;
        bool isIn = (sh.flags & 0x1u) != 0;
        if (fFilterIn && !isIn) continue;
        if (fFilterOut && isIn) continue;
        G4int idx = 0;
        (*fEvtMap)[idx] += (G4double)sh.photonWeight;
        fAccumulatedCount += (G4double)sh.photonWeight;
    }
}

void TsScoreGPUOpticalPhotonSurfaceTrackCount::UserHookForEndOfRun() {
    if (!fGPUEngine) return;

    fGPUEngine->FinalizePendingBatch();

    // SF1 deficit fix (2026-04-24): Run-lifetime accumulator 단일 read.
    // EndOfEvent의 cacheVer 처리 무시하고, accumulator에서 ALL hits 한번에 처리.
    // 다른 SF1 scorer가 같은 accumulator 공유 — 각자 surfaceId/filter로 분리.
    // EndOfRun cacheVer check: 마지막 batch가 EndOfEvent에서 처리 안 됐을 수 있음.
    uint64_t currentVersion = fGPUEngine->GetCacheVersion();
    if (currentVersion > fLastProcessedCacheVersion) {
        fLastProcessedCacheVersion = currentVersion;
        const auto& surfaceHits = fGPUEngine->GetCachedSurfaceHits();
        for (const auto& sh : surfaceHits) {
            if (sh.surfaceId != fSurfaceId) continue;
            bool isIn = (sh.flags & 0x1u) != 0;
            if (fFilterIn && !isIn) continue;
            if (fFilterOut && isIn) continue;
            G4int idx = 0;
            (*fEvtMap)[idx] += (G4double)sh.photonWeight;
            fAccumulatedCount += (G4double)sh.photonWeight;
        }
    }

    if (fEvtMap && !fEvtMap->empty()) {
        for (auto& kv : *fEvtMap) {
            G4int idx = kv.first;
            if (idx >= 0 && idx < (G4int)fFirstMomentMap.size()) {
                fFirstMomentMap[idx] += kv.second;
            }
        }
        fEvtMap->clear();
    }
}

void TsScoreGPUOpticalPhotonSurfaceTrackCount::Output() {
    std::string csvPath = fOutFileName + ".csv";
    FILE* fp = fopen(csvPath.c_str(), "w");
    if (!fp) {
        std::cerr << "[TsGPU-SF1] cannot open " << csvPath << std::endl;
        return;
    }
    fprintf(fp, "# TOPAS Version: %s\n", fPm->GetTOPASVersion().c_str());
    fprintf(fp, "# Parameter File: GPU Optical Photon Surface Track Count\n");
    fprintf(fp, "# Results for scorer: %s\n", GetName().c_str());
    fprintf(fp, "# Scored on surface: %s\n", fComponentName.c_str());
    fprintf(fp, "# Filter: %s\n", fFilterIn ? "going_in" : (fFilterOut ? "going_out" : "both"));
    fprintf(fp, "# GPUOpticalPhotonSurfaceTrackCount : Sum\n");
    G4double val = (fFirstMomentMap.size() > 0) ? fFirstMomentMap[0] : 0.0;
    fprintf(fp, "%.10g\n", val);
    fclose(fp);
    std::cout << "[TsGPU-SF1] Wrote " << csvPath << " count=" << val << std::endl;
}

void TsScoreGPUOpticalPhotonSurfaceTrackCount::Clear() {
    fFirstMomentMap.assign(fFirstMomentMap.size(), 0.0);
    if (fEvtMap) fEvtMap->clear();
    fAccumulatedCount = 0.0;
}
