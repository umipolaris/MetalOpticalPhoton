// Scorer for GPUOpticalPhotonPhaseSpace
/**
 * TsScoreGPUOpticalPhotonPhaseSpace.cc
 * TOPAS Extension Scorer: GPU 광학 광자 Phase-Space Dump
 *
 * 2026-05-19: 광자 별 phase-space record (transit hit) 를 ASCII CSV 로 export.
 * Multi-scorer wavelength binning 의 hidden limit 회피용 — 무한 fine wavelength
 * rainbow plot 가능 (광자 별 individual record + 후처리 임의 bin).
 *
 * Architecture: production code 의 s_accumulatedTransitHits (Run-lifetime
 * accumulator) 사용. HarvestPendingGPUResults 가 매 batch 마다 append, EndOfRun
 * 에서 1회 read → NumberOfHistoriesInRun 어느 값이어도 작동.
 *
 * Output CSV columns (per transit hit):
 *   photonIdx, posX_mm, posY_mm, posZ_mm, dirX, dirY, dirZ, pathLen_mm,
 *   energy_eV, wavelength_nm, photonWeight, volumeId
 *
 * Note: MOPHit encoding from MetalOpticalEngine.mm GetTransitHits():
 *   - posX/Y/Z = AABB entry point
 *   - time field = photonWeight (default 1.0)
 *   - flags field = pathLength (mm, geometric) reinterpret_cast<float>
 *   - _padding[0..2] = dirX/Y/Z (unit vector)
 */

#include "TsScoreGPUOpticalPhotonPhaseSpace.hh"
#include "TsGPUOpticalPhysics.hh"
#include "MetalOpticalEngine.h"  // MOPHit (typedef)

#include "TsParameterManager.hh"
#include "G4SystemOfUnits.hh"
#include "G4Event.hh"
#include "G4Run.hh"

#include <iostream>
#include <fstream>
#include <iomanip>

TsGPUOpticalPhysics* TsScoreGPUOpticalPhotonPhaseSpace::fGPUEngine = nullptr;

TsScoreGPUOpticalPhotonPhaseSpace::TsScoreGPUOpticalPhotonPhaseSpace(
    TsParameterManager* pM, TsMaterialManager* mM,
    TsGeometryManager* gM, TsScoringManager* scM,
    TsExtensionManager* eM,
    G4String scorerName, G4String quantity,
    G4String outFileName, G4bool isSubScorer)
    : TsVNtupleScorer(pM, mM, gM, scM, eM, scorerName, quantity, outFileName, isSubScorer)
    , fTotalRecords(0)
{
    fCsvPath = outFileName + ".phsp.csv";

    // 2026-05-23: PhaseSpace scorer 인스턴스가 존재할 때만 physics 가 transit hit 을
    // run-lifetime 누적하도록 enable (그 외 DDA Fluence-only 1B run 은 누적 skip → OOM 방지).
    TsGPUOpticalPhysics::s_transitAccumulationEnabled = true;

    std::cout << "[TsGPU-PhSp] PhaseSpace scorer '" << scorerName
              << "' initialized. Output: " << fCsvPath << std::endl;
}

TsScoreGPUOpticalPhotonPhaseSpace::~TsScoreGPUOpticalPhotonPhaseSpace() {}

void TsScoreGPUOpticalPhotonPhaseSpace::SetGPUEngine(TsGPUOpticalPhysics* engine) {
    fGPUEngine = engine;
}

G4bool TsScoreGPUOpticalPhotonPhaseSpace::ProcessHits(G4Step*, G4TouchableHistory*) {
    return false;
}

void TsScoreGPUOpticalPhotonPhaseSpace::UpdateForNewRun(G4bool rebuiltSomeComponents) {
    TsVNtupleScorer::UpdateForNewRun(rebuiltSomeComponents);
    if (fGPUEngine) {
        // SurfaceTrackCount scorer pattern: static flag 로 첫 scorer 만 reset.
        static bool s_accReset = false;
        if (!s_accReset) {
            fGPUEngine->ResetAccumulatedTransitHits();
            s_accReset = true;
        }
    }
}

void TsScoreGPUOpticalPhotonPhaseSpace::UserHookForEndOfEvent() {
    if (!fGPUEngine) return;
    // GPU propagation trigger (idempotent).
    // Accumulator append 는 production code 의 HarvestPendingGPUResults 가 자동
    // 수행 → 매 batch 마다 새 transit hits 가 s_accumulatedTransitHits 에 누적.
    fGPUEngine->EndOfEvent();
}

void TsScoreGPUOpticalPhotonPhaseSpace::UserHookForEndOfRun() {
    if (!fGPUEngine) {
        std::cerr << "[TsGPU-PhSp] WARNING: fGPUEngine null — no phsp dump" << std::endl;
        return;
    }

    // 잔여 batch flush + 최종 harvest (NumberOfHistoriesInRun % batchSize ≠ 0 시)
    fGPUEngine->FinalizePendingBatch();

    const std::vector<MOPHit>& hits = fGPUEngine->GetAccumulatedTransitHits();
    uint64_t total = hits.size();
    std::cout << "[TsGPU-PhSp] EndOfRun: total accumulated transit hits = "
              << total << std::endl;

    // 2026-05-19 perf fix: ostream 매 record format 매우 느림 (6M records 10s+).
    // snprintf + raw FILE* + large buffer (4 MB) → ~10x 빠름.
    FILE* f = std::fopen(fCsvPath.c_str(), "wb");
    if (!f) {
        std::cerr << "[TsGPU-PhSp] ERROR: cannot open " << fCsvPath << std::endl;
        return;
    }
    constexpr size_t IO_BUF_SIZE = 4 * 1024 * 1024;  // 4 MB
    char* ioBuf = (char*)std::malloc(IO_BUF_SIZE);
    std::setvbuf(f, ioBuf, _IOFBF, IO_BUF_SIZE);
    std::fputs("# GPU PhaseSpace record\n", f);
    std::fprintf(f, "# total records: %llu\n", (unsigned long long)total);
    std::fputs("# idx, posX_mm, posY_mm, posZ_mm, dirX, dirY, dirZ, pathLen_mm, energy_eV, wavelength_nm, photonWeight, volumeId\n", f);
    char line[512];
    for (uint64_t i = 0; i < total; i++) {
        const MOPHit& h = hits[i];
        // GetTransitHits encoding: flags=pathLen, time=photonWeight, _padding[0..2]=dirX/Y/Z
        float pathLen, dirX, dirY, dirZ;
        std::memcpy(&pathLen, &h.flags, sizeof(float));
        std::memcpy(&dirX, &h._padding[0], sizeof(float));
        std::memcpy(&dirY, &h._padding[1], sizeof(float));
        std::memcpy(&dirZ, &h._padding[2], sizeof(float));
        int n = std::snprintf(line, sizeof(line),
            "%llu, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %u\n",
            (unsigned long long)i,
            h.posX, h.posY, h.posZ,
            dirX, dirY, dirZ,
            pathLen,
            h.energy, h.wavelength, h.time,
            h.volumeId);
        std::fwrite(line, 1, (size_t)n, f);
    }
    std::fclose(f);
    std::free(ioBuf);
    fTotalRecords = total;
    std::cout << "[TsGPU-PhSp] Wrote " << total << " records to "
              << fCsvPath << std::endl;
}
