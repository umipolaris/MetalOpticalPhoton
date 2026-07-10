// Scorer for GPUOpticalPhotonCount
/**
 * TsScoreGPUOpticalPhotonCount.cc
 * TOPAS Extension Scorer: GPU 광학 광자 검출 결과를 CSV/ROOT로 출력
 *
 * GPU 엔진의 EndOfEvent() 후 히트 데이터를 수집하여 TOPAS Ntuple에 기록
 * 기존 TOPAS OpticalPhotonCount 스코어러와 동일한 출력 형식 제공
 */

#include "TsScoreGPUOpticalPhotonCount.hh"
#include "TsGPUOpticalPhysics.hh"

#include "TsParameterManager.hh"
#include "G4SystemOfUnits.hh"

#include <iostream>
#include <mutex>

// 정적 GPU 엔진 참조
TsGPUOpticalPhysics* TsScoreGPUOpticalPhotonCount::fGPUEngine = nullptr;

TsScoreGPUOpticalPhotonCount::TsScoreGPUOpticalPhotonCount(
    TsParameterManager* pM, TsMaterialManager* mM,
    TsGeometryManager* gM, TsScoringManager* scM,
    TsExtensionManager* eM,
    G4String scorerName, G4String quantity,
    G4String outFileName, G4bool isSubScorer)
    : TsVNtupleScorer(pM, mM, gM, scM, eM, scorerName, quantity, outFileName, isSubScorer)
{
    // Ntuple 컬럼 정의 (TsScoreOpticalPhotonCount 호환 형식)
    fNtuple->RegisterColumnF(&fPosX, "Position X", "cm");
    fNtuple->RegisterColumnF(&fPosY, "Position Y", "cm");
    fNtuple->RegisterColumnF(&fPosZ, "Position Z", "cm");
    fNtuple->RegisterColumnF(&fWaveLength, "Wavelength", "nm");
    fNtuple->RegisterColumnF(&fTime, "Arrival Time", "ns");
    fNtuple->RegisterColumnF(&fEnergy, "Energy", "eV");
    fNtuple->RegisterColumnI(&fVolumeId, "Volume ID");
    fNtuple->RegisterColumnI(&fParentTrackId, "Parent Track ID");
}

TsScoreGPUOpticalPhotonCount::~TsScoreGPUOpticalPhotonCount() {
}

void TsScoreGPUOpticalPhotonCount::SetGPUEngine(TsGPUOpticalPhysics* engine) {
    fGPUEngine = engine;
}

G4bool TsScoreGPUOpticalPhotonCount::ProcessHits(G4Step*, G4TouchableHistory*) {
    // GPU 모드에서는 ProcessHits가 호출되지 않음
    // (광학 광자는 kill 프로세스에 의해 즉시 제거되므로)
    // 대신 UserHookForEndOfEvent에서 GPU 히트를 수집
    return false;
}

void TsScoreGPUOpticalPhotonCount::UserHookForEndOfEvent() {
    if (!fGPUEngine) {
        // [주석처리 2026-05-14] guard 없는 초기화-단계 진단 출력
        // static int dbg = 0;
        // if (dbg++ < 3)
        //     std::cout << "[TsGPU-Score] UserHookForEndOfEvent: fGPUEngine is null!" << std::endl;
        return;
    }
    // 1. GPU 전파 실행 (누적된 genstep을 GPU에서 처리, idempotent)
    fGPUEngine->EndOfEvent();

    // 2. 캐시된 히트 데이터 사용 (MT 안전: 스레드 로컬 캐시)
    const auto& hits = fGPUEngine->GetCachedHits();
    uint32_t hitCount = (uint32_t)hits.size();

    if (hitCount > 0) {
        // 3. 각 히트를 Ntuple에 기록
        for (uint32_t i = 0; i < hitCount; i++) {
            const MOPHit& hit = hits[i];

            // GPU 단위 → Geant4 내부 단위 변환
            fPosX = (G4float)(hit.posX * CLHEP::mm);
            fPosY = (G4float)(hit.posY * CLHEP::mm);
            fPosZ = (G4float)(hit.posZ * CLHEP::mm);
            fWaveLength = (G4float)(hit.wavelength * CLHEP::nm);
            fTime = (G4float)(hit.time * CLHEP::ns);
            fEnergy = (G4float)(hit.energy * CLHEP::eV);
            fVolumeId = (G4int)(hit.volumeId);
            fParentTrackId = (G4int)(hit.parentTrackId);

            fNtuple->Fill();
        }
    }

    // 통계 출력 (100 이벤트마다 합산)
    static std::mutex logMutex;
    static uint32_t globalEventCount = 0;
    static uint32_t lastReportedAt = 0;

    {
        std::lock_guard<std::mutex> lock(logMutex);
        globalEventCount++;

        if (globalEventCount - lastReportedAt >= 100) {
            // [주석처리 2026-05-14] guard 없는 100-event 주기 진행 진단 출력
            // std::cout << "[TsGPU-Score] Events " << (lastReportedAt + 1) << "-" << globalEventCount
            //           << ": +" << globalIntervalHits << " hits"
            //           << " (total: " << globalTotalHits << " hits)" << std::endl;
            lastReportedAt = globalEventCount;
        }
    }
}

// Fix C1: Run 종료 시 잔여 배치 flush 후 남은 hit을 Ntuple에 추가
void TsScoreGPUOpticalPhotonCount::UserHookForEndOfRun() {
    if (!fGPUEngine) return;

    uint32_t nHits = fGPUEngine->FinalizePendingBatch();
    if (nHits == 0) return;

    const auto& hits = fGPUEngine->GetCachedHits();
    uint32_t hitCount = (uint32_t)hits.size();
    if (hitCount == 0) return;

    std::cout << "[TsGPU-Score] EndOfRun flush: +" << hitCount << " hits recorded" << std::endl;

    for (uint32_t i = 0; i < hitCount; i++) {
        const MOPHit& hit = hits[i];
        fPosX = (G4float)(hit.posX * CLHEP::mm);
        fPosY = (G4float)(hit.posY * CLHEP::mm);
        fPosZ = (G4float)(hit.posZ * CLHEP::mm);
        fWaveLength = (G4float)(hit.wavelength * CLHEP::nm);
        fTime = (G4float)(hit.time * CLHEP::ns);
        fEnergy = (G4float)(hit.energy * CLHEP::eV);
        fVolumeId = (G4int)(hit.volumeId);
        fParentTrackId = (G4int)(hit.parentTrackId);
        fNtuple->Fill();
    }
}
