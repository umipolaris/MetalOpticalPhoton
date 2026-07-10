// Scorer for GPUOpticalPhotonFluence
/**
 * TsScoreGPUOpticalPhotonFluence.cc
 * TOPAS Extension Scorer: GPU 광학 광자 Fluence 빈 스코어러
 *
 * Fix 41b: AABB 기반 구현
 * GPU 커널에서 매 전파 스텝마다 광자가 스코어링 AABB 내에 있으면
 * 진입점 위치 + AABB 내 경로 길이를 transit hit으로 기록
 * 스코어러에서 이 정보를 빈에 매핑하여 step_length/volume 누적
 *
 * CPU Fluence 스코어러와 동일한 물리량 측정:
 *   Fluence = Σ (step_length / bin_volume)
 */

#include "TsScoreGPUOpticalPhotonFluence.hh"
#include "TsGPUOpticalPhysics.hh"
// Fix 65: GPU DDA API — forward declaration
extern "C" {
void MOPEngine_RunGPUDDA(void* engine, float tx, float ty, float tz,
                          float fx, float fy, float fz,
                          uint32_t nx, uint32_t ny, uint32_t nz);
// F2 정석 fix (2026-04-23): scoringMaterialId 필터 버전
void MOPEngine_RunGPUDDA_WithMaterial(void* engine,
                                       float tx, float ty, float tz,
                                       float fx, float fy, float fz,
                                       uint32_t nx, uint32_t ny, uint32_t nz,
                                       uint32_t scoringMaterialId);
uint32_t MOPEngine_GetMaterialId(void* engine, const char* name);
/* 2026-05-12: uint32 → atomic_float (per-voxel cumulative overflow fix) */
const float* MOPEngine_GetDDABinBuffer(void* engine);
uint32_t MOPEngine_GetDDABinCount(void* engine);
float MOPEngine_GetDDAFluenceScale(void* engine);
// Phase 2.1+ (2026-05-16): cylindrical/spherical voxelization 지원 ext.
void MOPEngine_RunGPUDDA_AccumulateExt(void* engine, int scorerHandle,
                                        float tx, float ty, float tz,
                                        float fx, float fy, float fz,
                                        uint32_t nx, uint32_t ny, uint32_t nz,
                                        uint32_t scoringMaterialId,
                                        uint32_t voxelType,
                                        float rMin, float phiStart, float thetaStart);
}

#include "TsParameterManager.hh"
#include "TsVGeometryComponent.hh"
#include "G4SystemOfUnits.hh"
#include "G4VPhysicalVolume.hh"
#include "G4ParticleDefinition.hh"
#include "G4TransportationManager.hh"
#include "G4Tubs.hh"      // Phase 2.1+ (2026-05-16): cylindrical voxelization
#include "G4Sphere.hh"    // Phase 2.1+ (2026-05-16): spherical voxelization
#include "G4LogicalVolume.hh"

#include <iostream>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <set>
#include <mutex>
#include <string>

// 정적 GPU 엔진 참조
TsGPUOpticalPhysics* TsScoreGPUOpticalPhotonFluence::fGPUEngine = nullptr;

// Multi-scorer race fix (2026-05-01): 모든 scorer 인스턴스 공유 mutex.
// 자세한 이유는 헤더 sScorerMutex 주석 참고.
std::mutex TsScoreGPUOpticalPhotonFluence::sScorerMutex;
std::set<std::string> TsScoreGPUOpticalPhotonFluence::sScorerNamesWithRepropagate;
std::map<std::string, int> TsScoreGPUOpticalPhotonFluence::sScorerNameToHandle;
std::set<std::string> TsScoreGPUOpticalPhotonFluence::sScorerNamesWithCallback;
std::set<std::string> TsScoreGPUOpticalPhotonFluence::sAllScorerNames;
thread_local bool TsScoreGPUOpticalPhotonFluence::sPropagationTriggered = false;

G4bool TsScoreGPUOpticalPhotonFluence::fAABBRegisteredOnce = false;

TsScoreGPUOpticalPhotonFluence::TsScoreGPUOpticalPhotonFluence(
    TsParameterManager* pM, TsMaterialManager* mM,
    TsGeometryManager* gM, TsScoringManager* scM,
    TsExtensionManager* eM,
    G4String scorerName, G4String quantity,
    G4String outFileName, G4bool isSubScorer)
    : TsVBinnedScorer(pM, mM, gM, scM, eM, scorerName, quantity, outFileName, isSubScorer)
    , fCompTransX(0), fCompTransY(0), fCompTransZ(0)
    , fCompFullX(0), fCompFullY(0), fCompFullZ(0)
    , fNBinsX(1), fNBinsY(1), fNBinsZ(1)
    , fGeomCached(false)
    , fScoringVolumeRegistered(false)
{
    // 2026-05-28 multi-scorer fix: 전체 GPU optical scorer 이름 수집 (= 전체 scorer 수).
    // 첫 EndOfEvent 에서 모든 scorer 가 harvest 콜백을 등록한 뒤 마지막 scorer 가 1회
    // ForceRepropagate → 단일 propagation 의 transit hit 을 harvest 루프가 전 scorer 콜백에
    // 재사용(scorer 별 DDA). 과거: 첫 scorer 가 먼저 propagation 트리거 → 나머지 콜백 미등록 →
    // 2번째+ scorer 가 빈 buffer (0). (옛 Metal driver heap corruption 은 현재 빌드서 재현 안 됨.)
    std::string firstName;
    {
        std::lock_guard<std::mutex> lk(sScorerMutex);
        sAllScorerNames.insert(scorerName);
        if (sAllScorerNames.size() > 1) firstName = *sAllScorerNames.begin();
    }
    if (!firstName.empty()) {
        std::cerr << "[INFO] GPUOpticalPhotonFluence: multi-scorer 활성 (deferred-trigger 재사용) — '"
                  << firstName << "' + '" << scorerName << "'\n";
    }
    SetUnit("/mm2");
}

TsScoreGPUOpticalPhotonFluence::~TsScoreGPUOpticalPhotonFluence() {
    // 2026-05-12 cleanup: Fix54/PH 진단 출력 제거 (H1-H18 검증 종료, atomic_float +
    // harvest callback fix 적용 후 잔여 deficit 없음).
}

void TsScoreGPUOpticalPhotonFluence::SetGPUEngine(TsGPUOpticalPhysics* engine) {
    fGPUEngine = engine;
}

// Fix 70j: Output() 오버라이드 — Z-fastest CSV 직접 출력 (TOPAS 기본 대체)
void TsScoreGPUOpticalPhotonFluence::Output() {
    if (!fGeomCached) CacheComponentGeometry();
    if (!fGeomCached) return;

    G4double binSizeX = fCompFullX / fNBinsX;
    G4double binSizeY = fCompFullY / fNBinsY;
    G4double binSizeZ = fCompFullZ / fNBinsZ;

    std::string csvPath = fOutFileName + ".csv";
    FILE* fp = fopen(csvPath.c_str(), "w");
    if (!fp) {
        std::cerr << "[TsGPU-Fluence] Error: cannot open " << csvPath << std::endl;
        return;
    }

    // TOPAS CPU Fluence 호환 헤더
    fprintf(fp, "# TOPAS Version: %s\n", fPm->GetTOPASVersion().c_str());
    fprintf(fp, "# Parameter File: GPU Optical Photon Fluence\n");
    fprintf(fp, "# Results for scorer: %s\n", GetName().c_str());
    fprintf(fp, "# Filtered by: OnlyIncludeParticlesNamed = 1 \"opticalphoton\"\n");
    fprintf(fp, "# Scored in component: %s\n", fComponentName.c_str());
    fprintf(fp, "# X in %d bins of %.4f cm\n", fNBinsX, binSizeX / CLHEP::cm);
    fprintf(fp, "# Y in %d bins of %.4f cm\n", fNBinsY, binSizeY / CLHEP::cm);
    fprintf(fp, "# Z in %d bins of %.4f cm\n", fNBinsZ, binSizeZ / CLHEP::cm);

    if (fNBinsE > 1) {
        // 2026-05-20 EBins: CPU native XZE format 호환.
        //   row order = Z-outer X-inner (row = iZ*nX + iX), col = [underflow, nBinsE main, overflow, no-track].
        //   GPU buffer layout = [eBin*totalVoxels + voxel], voxel = iX*nY*nZ + iY*nZ + iZ.
        double eBinW = (fEBinMaxEv - fEBinMinEv) / fNBinsE;
        fprintf(fp, "# GPUOpticalPhotonFluence ( /mm2 ) : Sum   \n");
        fprintf(fp, "# Binned by incident track energy in %d bins of %.4g eV from %.4g eV to %.4g eV\n",
                fNBinsE, eBinW, fEBinMinEv, fEBinMaxEv);
        fprintf(fp, "# First bin is underflow, next to last bin is overflow, last bin is for case of no incident track.\n");
        uint32_t totalVoxels = (uint32_t)fNBinsX * fNBinsY * fNBinsZ;
        for (G4int iZ = 0; iZ < fNBinsZ; iZ++) {
            for (G4int iY = 0; iY < fNBinsY; iY++) {
                for (G4int iX = 0; iX < fNBinsX; iX++) {
                    uint32_t voxel = (uint32_t)iX * fNBinsY * fNBinsZ + (uint32_t)iY * fNBinsZ + (uint32_t)iZ;
                    fprintf(fp, "0");  // underflow col
                    for (G4int e = 0; e < fNBinsE; e++) {
                        uint32_t idx = (uint32_t)e * totalVoxels + voxel;
                        double val = (idx < fGpuFullBins.size()) ? fGpuFullBins[idx] : 0.0;
                        fprintf(fp, ", %.10g", val);
                    }
                    fprintf(fp, ", 0, 0\n");  // overflow, no-track
                }
            }
        }
        fclose(fp);
        if (fGPUEngine && fGPUEngine->GetLogLevel() >= 1) {
            std::cout << "[TsGPU-Fluence] X-Z-E CSV (nBinsE=" << fNBinsE << "): " << csvPath << std::endl;
        }
        return;
    }

    fprintf(fp, "# GPUOpticalPhotonFluence ( /mm2 ) : Sum   \n");

    // 2026-06-04 FIX: 행 순서를 CPU(TsScoreFluence)와 일치 = iX-fastest, iZ-outer.
    //   이전엔 iX-outer/iZ-fastest 로 써서 CPU CSV 와 transpose 되어 있었음(EBins>1 경로는
    //   이미 iZ-outer 라 일치했으나 단일-E 경로만 어긋남). naive row-order 비교가
    //   GPU 공간분포를 scramble → 가짜 deficit 유발하던 confound 제거. idx/인덱스 출력은 불변.
    for (G4int iZ = 0; iZ < fNBinsZ; iZ++) {
        for (G4int iY = 0; iY < fNBinsY; iY++) {
            for (G4int iX = 0; iX < fNBinsX; iX++) {
                G4int idx = iX * fNBinsY * fNBinsZ + iY * fNBinsZ + iZ;
                G4double val = (idx < (G4int)fFirstMomentMap.size()) ? fFirstMomentMap[idx] : 0.0;
                fprintf(fp, "%d, %d, %d, %.10g\n", iX, iY, iZ, val);
            }
        }
    }

    fclose(fp);
    if (fGPUEngine && fGPUEngine->GetLogLevel() >= 1) {
        std::cout << "[Fix70j] Z-fastest CSV: " << csvPath << std::endl;
    }

    // GPU 성능 통계 출력
    if (fGPUEngine && fGPUEngine->GetLogLevel() >= 1) {
        MOPPropagationStats stats;
        fGPUEngine->GetStats(&stats);
        std::cout << "[GPU-Perf] ======== GPU Performance ========" << std::endl;
        std::cout << "[GPU-Perf]   Photons generated: " << stats.totalPhotonsGenerated << std::endl;
        std::cout << "[GPU-Perf]   Photons detected:  " << stats.totalPhotonsDetected << std::endl;
        std::cout << "[GPU-Perf]   GPU time:          " << stats.gpuTimeMs << " ms (wall)" << std::endl;
        std::cout << "[GPU-Perf]   Total time:        " << stats.totalTimeMs << " ms" << std::endl;
        // Fix 78측정: HW 카운터 기반 dispatch별 GPU 시간 분리
        double total = stats.gpuPropagateMs + stats.gpuDDAMs;
        if (total > 0.0) {
            std::cout << "[GPU-Perf] --- HW counter breakdown ---" << std::endl;
            std::cout << "[GPU-Perf]   propagate GPU: " << stats.gpuPropagateMs << " ms ("
                      << (100.0 * stats.gpuPropagateMs / total) << "%) ["
                      << stats.propagateDispatchCount << " dispatches]" << std::endl;
            std::cout << "[GPU-Perf]   DDA       GPU: " << stats.gpuDDAMs << " ms ("
                      << (100.0 * stats.gpuDDAMs / total) << "%) ["
                      << stats.ddaDispatchCount << " dispatches]" << std::endl;
            std::cout << "[GPU-Perf]   sum dispatch:  " << total << " ms" << std::endl;
        }
        // Fix 78측정 2단계: per-photon / per-transit cost 분석
        if (stats.totalPhotonsPropagated > 0 && stats.gpuPropagateMs > 0) {
            double nsPerPhoton = (stats.gpuPropagateMs * 1e6) / (double)stats.totalPhotonsPropagated;
            double avgPerDispatch = stats.gpuPropagateMs / std::max(1u, stats.propagateDispatchCount);
            double photonsPerDispatch = (double)stats.totalPhotonsPropagated / std::max(1u, stats.propagateDispatchCount);
            std::cout << "[GPU-Perf] --- propagate kernel cost ---" << std::endl;
            std::cout << "[GPU-Perf]   total photons:    " << stats.totalPhotonsPropagated << std::endl;
            std::cout << "[GPU-Perf]   per-photon cost:  " << nsPerPhoton << " ns" << std::endl;
            std::cout << "[GPU-Perf]   per-dispatch:     " << avgPerDispatch << " ms ("
                      << photonsPerDispatch << " photons)" << std::endl;
            std::cout << "[GPU-Perf]   transit hits:     " << stats.totalTransitHitsRecorded
                      << " (" << ((double)stats.totalTransitHitsRecorded / (double)stats.totalPhotonsPropagated * 100.0)
                      << "% of photons)" << std::endl;
        }
        std::cout << "[GPU-Perf] ===================================" << std::endl;
    }
}

void TsScoreGPUOpticalPhotonFluence::Clear() {
    fFirstMomentMap.assign(fFirstMomentMap.size(), 0.0);
    fEvtMap->clear();
}

// Fix C1: Run 종료 훅 — 잔여 배치 flush + 마지막 transit hit을 fEvtMap에 반영
// (EventBatchSize로 인해 N%BatchSize≠0 이벤트가 배치 버퍼에 남아 있는 경우)
void TsScoreGPUOpticalPhotonFluence::UserHookForEndOfRun() {
    if (!fGPUEngine) return;
    // Fix 78측정-2 lean: hook-wide lock 제거. Wrapper 들이 m_gpuMutex 알아서.
    if (!fGeomCached) CacheComponentGeometry();
    if (!fGeomCached) return;

    // 잔여 배치 dispatch + 비동기 결과 수확
    uint32_t flushedHits = fGPUEngine->FinalizePendingBatch();

    const auto& transitHits = fGPUEngine->GetCachedTransitHits();
    if (transitHits.empty()) {
        if (flushedHits > 0 && fGPUEngine->GetLogLevel() >= 1) {
            std::cout << "[TsGPU-Fluence] EndOfRun: "
                      << flushedHits << " hits but no transit hits" << std::endl;
        }
        // Multi-scorer race fix follow-up (2026-05-02): tail rescue path 가 비어 있어도
        // per-scorer accumulating GPU buffer 는 모든 batch 결과를 가지고 있으므로 transfer
        // 는 반드시 수행해야 함. 이전 early return 이 transfer 를 skip → CSV 0 byte.
        if (fScorerBufferHandle >= 0 && fGPUEngine) {
            uint32_t binCount = 0;
            // 2026-05-12: uint32 → atomic_float (per-voxel cumulative overflow fix)
            // fluenceScale = 1.0f (float 누적이므로 scaling 불필요)
            const float* gpuBins = fGPUEngine->GetScorerBinBufferLocked(fScorerBufferHandle, &binCount);
            if (gpuBins && binCount > 0) {
                const uint32_t totalBins = (uint32_t)fFirstMomentMap.size();
                uint32_t lim = std::min(binCount, totalBins);
                G4double scorerSum = 0;
                for (uint32_t i = 0; i < lim; i++) {
                    if (gpuBins[i] > 0.0f) {
                        G4double val = (G4double)gpuBins[i];
                        fFirstMomentMap[i] += val;
                        scorerSum += val;
                    }
                }
                if (fGPUEngine && fGPUEngine->GetLogLevel() >= 1) {
                    std::cout << "[TsGPU-Fluence] EndOfRun GPU buffer transfer (no transit-tail): handle="
                              << fScorerBufferHandle << " bins=" << lim
                              << " sum=" << scorerSum << std::endl;
                }
            }
        }
        return;
    }

    if (fGPUEngine->GetLogLevel() >= 1) {
        std::cout << "[TsGPU-Fluence] EndOfRun flush: "
                  << transitHits.size() << " residual transit hits recorded "
                  << "(tail batch rescue)" << std::endl;
    }

    // UserHookForEndOfEvent와 동일한 DDA 경로: CPU DDA만 사용
    // (GPU DDA 경로는 fEvtMap에 이미 배치 시점에서 반영됨. 여기서는 잔여 transit hit을
    //  CPU DDA로 빈에 분배하여 누락 방지.)
    G4double eventRawPathMM = 0;

    // 2026-05-13 fix: tail rescue 의 CPU DDA path 가 scorer buffer transfer (line
    // 331-351) 와 double-count → 광자 1개당 fluence 2× over.
    // Harvest callback (RegisterHarvestCallback) 이 FinalizePendingBatch 의
    // HarvestPendingGPUResults 시점에 GPU DDA 가 모든 batch 의 transit hits 를
    // scorer buffer 에 누적. 잔여 batch 도 동일 path 처리.
    // 따라서 여기 CPU DDA path 는 진단 카운트만 유지하고 fEvtMap 누적 제거.
    (void)eventRawPathMM;

    // Fix C1: fEvtMap → fFirstMomentMap 수동 전송 (legacy CPU 경로 호환)
    if (fEvtMap && !fEvtMap->empty()) {
        for (auto& kv : *fEvtMap) {
            G4int idx = kv.first;
            if (idx >= 0 && idx < (G4int)fFirstMomentMap.size()) {
                fFirstMomentMap[idx] += kv.second;
            }
        }
        fEvtMap->clear();
    }

    // Multi-scorer race fix (2026-05-01): per-scorer accumulating GPU buffer 한 번 read
    // 후 fFirstMomentMap 으로 transfer. per-event CPU read/iterate 를 EndOfRun 1회로
    // 대체 → CPU 동기화 부담 제거.
    if (fScorerBufferHandle >= 0 && fGPUEngine) {
        uint32_t binCount = 0;
        // 2026-05-12: uint32 → atomic_float (per-voxel cumulative overflow fix)
        // fluenceScale = 1.0f (float 누적이므로 scaling 불필요)
        const float* gpuBins = fGPUEngine->GetScorerBinBufferLocked(fScorerBufferHandle, &binCount);
        if (gpuBins && binCount > 0) {
            if (fNBinsE > 1) {
                // 2026-05-20 EBins: full buffer (voxel × nBinsE) → fGpuFullBins (Output X-Z-E).
                fGpuFullBins.assign(binCount, 0.0);
                G4double scorerSum = 0;
                for (uint32_t i = 0; i < binCount; i++) {
                    fGpuFullBins[i] = (G4double)gpuBins[i];
                    scorerSum += (G4double)gpuBins[i];
                }
                if (fGPUEngine && fGPUEngine->GetLogLevel() >= 1) {
                    std::cout << "[TsGPU-Fluence] EndOfRun GPU EBins buffer: bins=" << binCount
                              << " (voxel×nBinsE) sum=" << scorerSum << std::endl;
                }
            } else {
                const uint32_t totalBins = (uint32_t)fFirstMomentMap.size();
                uint32_t lim = std::min(binCount, totalBins);
                G4double scorerSum = 0;
                for (uint32_t i = 0; i < lim; i++) {
                    if (gpuBins[i] > 0.0f) {
                        G4double val = (G4double)gpuBins[i];
                        fFirstMomentMap[i] += val;
                        scorerSum += val;
                    }
                }
                if (fGPUEngine && fGPUEngine->GetLogLevel() >= 1) {
                    std::cout << "[TsGPU-Fluence] EndOfRun GPU buffer transfer: handle="
                              << fScorerBufferHandle << " bins=" << lim
                              << " sum=" << scorerSum << std::endl;
                }
            }
        }
    }
}

void TsScoreGPUOpticalPhotonFluence::CacheComponentGeometry() {
    if (fGeomCached) return;

    // 빈 수와 전체 크기를 컴포넌트에서 가져옴
    fNBinsX = fComponent->GetDivisionCount(0);
    fNBinsY = fComponent->GetDivisionCount(1);
    fNBinsZ = fComponent->GetDivisionCount(2);

    // 2026-05-15: fp32 ULP saturation 은 binning 만으로 결정 안 됨 (광자 집중도
    //   따라 thin tube 도 saturate 가능, 1×1×1500 도 5% off). static bin-count
    //   ban 대신 **런타임 감지** 사용 — DDAScoring.metal 이 add 가 ULP 밑이라
    //   lost 됐을 때 saturationFlag set, MetalOpticalEngine::GetScorerBinBuffer
    //   에서 flag 읽고 abort. 어떤 bin 구성에서든 실제 saturation 발생 시 잡힘.
    //   참조: memory [[u5scan_gpu_singlebin_fp32_saturation_2026_05_15]]

    fCompFullX = fComponent->GetFullWidth(0);  // Geant4 내부 단위 (mm 또는 rad)
    fCompFullY = fComponent->GetFullWidth(1);
    fCompFullZ = fComponent->GetFullWidth(2);

    // Phase 2.1+ (2026-05-16): component solid 판정 + cylindrical/spherical
    // voxelization 정보 캐시.  TOPAS 의 cylindrical/spherical component 는
    // GetFullWidth 가 (R width, Phi rad, Z mm) / (R width, Theta rad, Phi rad) 반환.
    // GPU shader 는 voxelType + rMin + phiStart + thetaStart 로 Cartesian↔cyl/sph
    // 변환 후 bin index 결정.
    fVoxelType = 0;
    fRMin = 0;
    fPhiStart = 0;
    fThetaStart = 0;
    if (fComponent) {
        G4LogicalVolume* lv = fComponent->GetEnvelopeLogicalVolume();
        G4VSolid* solid = lv ? lv->GetSolid() : nullptr;
        if (auto* tubs = dynamic_cast<G4Tubs*>(solid)) {
            fVoxelType = 1;   // CYLINDER
            fRMin     = tubs->GetInnerRadius();
            fPhiStart = tubs->GetStartPhiAngle();
        } else if (auto* sph = dynamic_cast<G4Sphere*>(solid)) {
            fVoxelType = 2;   // SPHERE
            fRMin       = sph->GetInnerRadius();
            fPhiStart   = sph->GetStartPhiAngle();
            fThetaStart = sph->GetStartThetaAngle();
        }
        if (fGPUEngine && fGPUEngine->GetLogLevel() >= 1) {
            std::cout << "[TsGPU-Fluence] voxelType=" << fVoxelType
                      << " (0=BOX 1=CYL 2=SPH) RMin=" << fRMin/CLHEP::mm
                      << "mm PhiStart=" << fPhiStart
                      << " ThetaStart=" << fThetaStart << std::endl;
        }
    }

    // TOPAS 파라미터로 월드 위치 계산
    {
        G4String compName = fComponentName;
        // 빈닝 접미사(_NxNxN) 제거
        size_t lastUnderscore = compName.rfind('_');
        if (lastUnderscore != std::string::npos) {
            G4String suffix = compName.substr(lastUnderscore + 1);
            if (suffix.find('x') != std::string::npos) {
                G4String originalName = compName.substr(0, lastUnderscore);
                if (fPm->ParameterExists("Ge/" + originalName + "/Parent")) {
                    compName = originalName;
                }
            }
        }

        G4double tx = 0, ty = 0, tz = 0;
        while (true) {
            G4String prefix = "Ge/" + compName + "/";

            if (fPm->ParameterExists(prefix + "TransX"))
                tx += fPm->GetDoubleParameter(prefix + "TransX", "Length");
            if (fPm->ParameterExists(prefix + "TransY"))
                ty += fPm->GetDoubleParameter(prefix + "TransY", "Length");
            if (fPm->ParameterExists(prefix + "TransZ"))
                tz += fPm->GetDoubleParameter(prefix + "TransZ", "Length");

            G4String parentParam = prefix + "Parent";
            if (!fPm->ParameterExists(parentParam)) break;

            compName = fPm->GetStringParameter(parentParam);
            if (compName == "World" || compName == "world") break;
        }

        fCompTransX = tx;
        fCompTransY = ty;
        fCompTransZ = tz;
    }

    fGeomCached = true;

    // Fix 78측정-3: 시나리오 자동 batch 튜닝
    // 사용자가 EventBatchSize 명시했으면 무시. 안 했으면 bin 수에 따라 적용.
    if (fGPUEngine) {
        uint64_t totalBins = (uint64_t)fNBinsX * fNBinsY * fNBinsZ;
        bool wasExplicit = fGPUEngine->IsBatchExplicit();
        int oldBatch = fGPUEngine->GetEventBatchSize();
        fGPUEngine->MaybeAutoTuneBatch(totalBins);
        int newBatch = fGPUEngine->GetEventBatchSize();
        if (!wasExplicit && newBatch != oldBatch && fGPUEngine->GetLogLevel() >= 1) {
            std::cout << "[TsGPU-Fluence] Auto batch tune: " << oldBatch << " → "
                      << newBatch << " (totalBins=" << totalBins
                      << ", " << (totalBins > 10000 ? "high-res/DDA" : "low-res") << ")"
                      << std::endl;
        }
    }

    // ABSOLUTE RULE (2026-05-04): GPU optical run 은 single-PW 강제.
    // Multi-PW (mass world 외 IsParallel=True 컴포넌트가 2 개 이상) 시
    // G4PathFinder 가 step 마다 모든 PW 추적 → CPU 50% navigation overhead.
    // GPU 는 자체 propagation 이라 PW navigation 자체가 무의미 — fatal abort.
    if (fGPUEngine && !fScoringVolumeRegistered) {
        auto* tm = G4TransportationManager::GetTransportationManager();
        size_t nNav = tm ? tm->GetNoActiveNavigators() : 0;
        if (nNav > 2) {
            std::cerr << "\n========================================================\n"
                      << "[FATAL] GPUOpticalPhotonFluence: multi-PW NOT SUPPORTED\n"
                      << "  Active navigators (mass + PWs) = " << nNav << " (>2 means >1 PW)\n"
                      << "  Allowed: ONE parallel world per run (1 GPU scorer = 1 PW).\n"
                      << "  Reason: extra PWs trigger G4PathFinder navigation on every\n"
                      << "  step (~50% CPU overhead) and are useless for GPU runs.\n"
                      << "========================================================\n"
                      << std::endl;
            std::abort();
        }
    }

    // Fix 41b: GPU에 스코어링 AABB 등록
    // Fix 43: 딸 볼륨(SensorSilicon 등) 제외 — CPU ProcessHits는
    //         부모 볼륨 재질(Air)만 스코어링하므로 GPU도 동일하게
    if (fGPUEngine && !fScoringVolumeRegistered) {
        // Phase 2.1+ (2026-05-16): cylindrical/spherical 의 cartesian AABB =
        // 외접 box (RMax 기준). CYL: (±RMax, ±RMax, ±HL).  SPH: (±RMax)³.
        // GPU 의 photon transit hit recording 은 cartesian AABB 안 광자만.
        G4double halfX, halfY, halfZ;
        if (fVoxelType == 1) {            // CYLINDER
            G4double rMax = fRMin + fCompFullX;
            halfX = halfY = rMax;
            halfZ = fCompFullZ * 0.5;
        } else if (fVoxelType == 2) {     // SPHERE
            G4double rMax = fRMin + fCompFullX;
            halfX = halfY = halfZ = rMax;
        } else {                           // BOX
            halfX = fCompFullX * 0.5;
            halfY = fCompFullY * 0.5;
            halfZ = fCompFullZ * 0.5;
        }

        // 딸 볼륨이 AABB 경계에 있는지 확인하여 AABB 축소
        // TOPAS 파라미터에서 딸 볼륨 정보 검색
        G4String compName = fComponentName;
        // 빈닝 접미사(_NxNxN) 제거
        size_t lastUnderscore = compName.rfind('_');
        if (lastUnderscore != std::string::npos) {
            G4String suffix = compName.substr(lastUnderscore + 1);
            if (suffix.find('x') != std::string::npos) {
                G4String originalName = compName.substr(0, lastUnderscore);
                if (fPm->ParameterExists("Ge/" + originalName + "/Parent")) {
                    compName = originalName;
                }
            }
        }

        // Fix 48: AABB는 전체 ImagingSensor를 커버 (Fix 43 제거)
        // Fix 43은 SensorSilicon 딸 볼륨을 제외했지만, 이로 인해
        // Z+ 방향에서 진입하는 광자의 transit hit이 누락됨.
        // CPU parallel world는 딸 볼륨 포함 전체 영역을 스코어링하므로
        // GPU AABB도 동일하게 전체 영역을 커버해야 함.

        // GPU는 mm 단위 사용 (Geant4 내부 단위도 mm)
        float minX = (float)(fCompTransX - halfX);
        float minY = (float)(fCompTransY - halfY);
        float minZ = (float)(fCompTransZ - halfZ);
        float maxX = (float)(fCompTransX + halfX);
        float maxY = (float)(fCompTransY + halfY);
        float maxZ = (float)(fCompTransZ + halfZ);  // Fix 48: 전체 볼륨 포함

        fGPUEngine->SetScoringAABB(minX, minY, minZ, maxX, maxY, maxZ);
        fScoringVolumeRegistered = true;

        // Multi-scorer race fix (2026-05-02 v2): scorer NAME 1개당 buffer 1개 — TOPAS
        // multi-thread 에서 worker 별 scorer instance 모두 같은 buffer 공유 (GPU atomic).
        // 2026-05-20: EBins (energy/wavelength binning) parse — CPU Fluence 동등.
        fNBinsE = 1; fEBinMinEv = 0.0; fEBinMaxEv = 0.0;
        if (fPm->ParameterExists(GetFullParmName("EBins"))) {
            fNBinsE = fPm->GetIntegerParameter(GetFullParmName("EBins"));
            if (fNBinsE < 1) fNBinsE = 1;
            if (fPm->ParameterExists(GetFullParmName("EBinMin")))
                fEBinMinEv = fPm->GetDoubleParameter(GetFullParmName("EBinMin"), "Energy") / eV;
            if (fPm->ParameterExists(GetFullParmName("EBinMax")))
                fEBinMaxEv = fPm->GetDoubleParameter(GetFullParmName("EBinMax"), "Energy") / eV;
        }
        if (fScorerBufferHandle < 0) {
            uint32_t totalBins = (uint32_t)fNBinsX * fNBinsY * fNBinsZ;
            std::lock_guard<std::mutex> lk(sScorerMutex);
            auto it = sScorerNameToHandle.find(GetName());
            if (it != sScorerNameToHandle.end()) {
                fScorerBufferHandle = it->second;
            } else {
                fScorerBufferHandle = fGPUEngine->RegisterScorerBinBufferLocked(
                    totalBins, (uint32_t)fNBinsE, (float)fEBinMinEv, (float)fEBinMaxEv);
                sScorerNameToHandle[GetName()] = fScorerBufferHandle;
                if (fGPUEngine->GetLogLevel() >= 1) {
                    std::cout << "[TsGPU-Fluence] Registered scorer buffer handle="
                              << fScorerBufferHandle << " totalBins=" << totalBins
                              << " nBinsE=" << fNBinsE << " E=[" << fEBinMinEv << "," << fEBinMaxEv << "]eV"
                              << " name=" << GetName() << std::endl;
                }
                // 2026-05-19 multi-scorer photon energy filter parse + 등록
                float eLow = 0.0f, eHigh = 0.0f;
                G4String parmAbove = GetFullParmName("OnlyIncludeParticlesWithPrimaryKEAbove");
                G4String parmBelow = GetFullParmName("OnlyIncludeParticlesWithPrimaryKEBelow");
                if (fPm->ParameterExists(parmAbove))
                    eLow = (float)(fPm->GetDoubleParameter(parmAbove, "Energy") / eV);
                if (fPm->ParameterExists(parmBelow))
                    eHigh = (float)(fPm->GetDoubleParameter(parmBelow, "Energy") / eV);
                if (eLow > 0 || eHigh > 0) {
                    fGPUEngine->SetScorerEnergyFilterLocked(fScorerBufferHandle, eLow, eHigh);
                }
            }
        }

        if (fGPUEngine->GetLogLevel() >= 1) {
            std::cout << "[TsGPU-Fluence] Scoring AABB registered (Fix 48 — full volume): "
                      << "(" << minX << "," << minY << "," << minZ << ") - "
                      << "(" << maxX << "," << maxY << "," << maxZ << ")"
                      << std::endl;
        }
    }

    if (fGPUEngine && fGPUEngine->GetLogLevel() >= 1) {
        std::cout << "[TsGPU-Fluence] Component '" << fComponentName << "'"
                  << " pos=(" << fCompTransX/mm << ", " << fCompTransY/mm << ", " << fCompTransZ/mm << ") mm"
                  << " size=(" << fCompFullX/mm << ", " << fCompFullY/mm << ", " << fCompFullZ/mm << ") mm"
                  << " bins=(" << fNBinsX << ", " << fNBinsY << ", " << fNBinsZ << ")"
                  << std::endl;
    }

    // 2026-05-12 fix: harvest callback 등록 (scorer name 기준 1회). HarvestPendingGPUResults 가
    // batch N 의 데이터를 cache 한 직후 (= batch N+1 propagation 이 dispatch 되기 전) 호출.
    // callback 안에서 DDA dispatch → queue FIFO 순서로 DDA 가 propagation 보다 먼저
    // 실행 → m_transitHitBuffer 에서 batch N 의 데이터 정확히 read.
    bool needCallbackReg = false;
    if (fGPUEngine && fScorerBufferHandle >= 0) {
        std::lock_guard<std::mutex> _cbLock(sScorerMutex);
        needCallbackReg = sScorerNamesWithCallback.insert(GetName()).second;
    }
    if (needCallbackReg) {
        fGPUEngine->RegisterHarvestCallback([this]() {
            if (!fGPUEngine || fScorerBufferHandle < 0) return;
            // Component material id (scorer material filter)
            uint32_t scorerMatId = 0xFFFFFFFFu;
            G4String matNameParam = "Ge/" + fComponentName + "/Material";
            if (fPm->ParameterExists(matNameParam)) {
                G4String matName = fPm->GetStringParameter(matNameParam);
                uint32_t mid = MOPEngine_GetMaterialId(fGPUEngine->GetEngine(), matName.c_str());
                if (mid != 0xFFFFFFFFu) scorerMatId = mid;
            }
            // DDA dispatch — m_lastTransitCount 와 m_transitHitBuffer 모두
            // 현재 harvest 된 batch 의 데이터. RunGPUDDAAccumulateLocked 는
            // recursive_mutex 안에서 inner Harvest skip (m_gpuPipelinePending=false).
            // Phase 2.1+ (2026-05-16): voxelType + cyl/sph extra info 전달.
            // CYL/SPH 인 경우 GetFullWidth(1/2) 가 이미 radian — mm 변환 안 함.
            float fullX = (float)(fCompFullX / CLHEP::mm);
            float fullY = (fVoxelType == 0) ? (float)(fCompFullY / CLHEP::mm)
                                            : (float)fCompFullY;   // rad
            float fullZ = (fVoxelType == 2) ? (float)fCompFullZ     // rad (SPH phi)
                                            : (float)(fCompFullZ / CLHEP::mm);
            fGPUEngine->RunGPUDDAAccumulateExtLocked(fScorerBufferHandle,
                (float)(fCompTransX / CLHEP::mm),
                (float)(fCompTransY / CLHEP::mm),
                (float)(fCompTransZ / CLHEP::mm),
                fullX, fullY, fullZ,
                fNBinsX, fNBinsY, fNBinsZ,
                scorerMatId,
                (uint32_t)fVoxelType,
                (float)(fRMin / CLHEP::mm),
                (float)fPhiStart,
                (float)fThetaStart);
        });
        if (fGPUEngine && fGPUEngine->GetLogLevel() >= 1) {
            std::cout << "[TsGPU-Fluence] Harvest callback registered" << std::endl;
        }
    }
}

G4bool TsScoreGPUOpticalPhotonFluence::ProcessHits(G4Step* /*aStep*/, G4TouchableHistory*) {
    // Fix 50: GPU scorer의 ProcessHits는 항상 false 반환
    // 광학 광자 Fluence는 GPU transit hit으로만 기록 (UserHookForEndOfEvent에서 처리)
    // ProcessHits에서 광학 광자를 AccumulateHit하면 validation mode에서
    // CPU 추적 광자가 fEvtMap에 이중 카운트됨 (GPU transit + CPU ProcessHits)
    //
    // 정상 모드: 광학 광자가 kill되어 ProcessHits 미호출 → 영향 없음
    // Validation 모드: CPU 광학 광자가 살아있어 ProcessHits 호출 → 이중 카운트 발생
    //
    // 따라서 ProcessHits는 어떤 입자도 스코어링하지 않음
    return false;
}

// ============================================================
// DDA Ray Tracing: transit hit의 광선을 빈 격자를 통해 추적
// ============================================================
void TsScoreGPUOpticalPhotonFluence::TraceRayThroughGrid(
    G4double startX, G4double startY, G4double startZ,
    G4double dirX, G4double dirY, G4double dirZ,
    G4double pathLength, G4double photonWeight)
{
    // 컴포넌트 로컬 좌표로 변환
    G4double localX = startX - fCompTransX;
    G4double localY = startY - fCompTransY;
    G4double localZ = startZ - fCompTransZ;

    G4double halfX = fCompFullX * 0.5;
    G4double halfY = fCompFullY * 0.5;
    G4double halfZ = fCompFullZ * 0.5;

    // 빈 크기
    G4double binSizeX = fCompFullX / fNBinsX;
    G4double binSizeY = fCompFullY / fNBinsY;
    G4double binSizeZ = fCompFullZ / fNBinsZ;
    G4double binVolume = binSizeX * binSizeY * binSizeZ;

    // 방향 벡터 정규화
    G4double dirLen = std::sqrt(dirX*dirX + dirY*dirY + dirZ*dirZ);
    if (dirLen < 1e-10) {
        // 방향이 거의 0인 경우: 진입점 위치의 빈에 pathLength만 기록
        G4int iX = (G4int)((localX + halfX) / binSizeX);
        G4int iY = (G4int)((localY + halfY) / binSizeY);
        G4int iZ = (G4int)((localZ + halfZ) / binSizeZ);
        iX = std::max(0, std::min(iX, fNBinsX - 1));
        iY = std::max(0, std::min(iY, fNBinsY - 1));
        iZ = std::max(0, std::min(iZ, fNBinsZ - 1));
        G4int binIndex = fComponent->GetIndex(iX, iY, iZ);
        if (binIndex >= 0) {
            G4double contrib = pathLength * photonWeight / binVolume;
            (*fEvtMap)[binIndex] += contrib;
        }
        return;
    }
    dirX /= dirLen;
    dirY /= dirLen;
    dirZ /= dirLen;

    // 광선과 컴포넌트 AABB의 교차점 계산
    G4double tmin = 0, tmax = pathLength;

    // X축
    if (std::abs(dirX) > 1e-10) {
        G4double t1 = (-halfX - localX) / dirX;
        G4double t2 = ( halfX - localX) / dirX;
        if (t1 > t2) std::swap(t1, t2);
        tmin = std::max(tmin, t1);
        tmax = std::min(tmax, t2);
    } else {
        if (localX < -halfX || localX > halfX) return;
    }

    // Y축
    if (std::abs(dirY) > 1e-10) {
        G4double t1 = (-halfY - localY) / dirY;
        G4double t2 = ( halfY - localY) / dirY;
        if (t1 > t2) std::swap(t1, t2);
        tmin = std::max(tmin, t1);
        tmax = std::min(tmax, t2);
    } else {
        if (localY < -halfY || localY > halfY) return;
    }

    // Z축
    if (std::abs(dirZ) > 1e-10) {
        G4double t1 = (-halfZ - localZ) / dirZ;
        G4double t2 = ( halfZ - localZ) / dirZ;
        if (t1 > t2) std::swap(t1, t2);
        tmin = std::max(tmin, t1);
        tmax = std::min(tmax, t2);
    } else {
        if (localZ < -halfZ || localZ > halfZ) return;
    }

    if (tmin >= tmax) return;

    // 진입점의 빈 인덱스
    G4double entryX = localX + dirX * tmin;
    G4double entryY = localY + dirY * tmin;
    G4double entryZ = localZ + dirZ * tmin;

    G4int iX = (G4int)((entryX + halfX) / binSizeX);
    G4int iY = (G4int)((entryY + halfY) / binSizeY);
    G4int iZ = (G4int)((entryZ + halfZ) / binSizeZ);

    iX = std::max(0, std::min(iX, fNBinsX - 1));
    iY = std::max(0, std::min(iY, fNBinsY - 1));
    iZ = std::max(0, std::min(iZ, fNBinsZ - 1));

    // DDA 스텝 방향
    G4int stepIX = (dirX >= 0) ? 1 : -1;
    G4int stepIY = (dirY >= 0) ? 1 : -1;
    G4int stepIZ = (dirZ >= 0) ? 1 : -1;

    G4double tMaxX, tMaxY, tMaxZ;
    G4double tDeltaX, tDeltaY, tDeltaZ;

    if (std::abs(dirX) > 1e-10) {
        G4double nextBoundX = -halfX + (iX + (stepIX > 0 ? 1 : 0)) * binSizeX;
        tMaxX = tmin + (nextBoundX - entryX) / dirX;
        tDeltaX = binSizeX / std::abs(dirX);
    } else {
        tMaxX = 1e30;
        tDeltaX = 1e30;
    }

    if (std::abs(dirY) > 1e-10) {
        G4double nextBoundY = -halfY + (iY + (stepIY > 0 ? 1 : 0)) * binSizeY;
        tMaxY = tmin + (nextBoundY - entryY) / dirY;
        tDeltaY = binSizeY / std::abs(dirY);
    } else {
        tMaxY = 1e30;
        tDeltaY = 1e30;
    }

    if (std::abs(dirZ) > 1e-10) {
        G4double nextBoundZ = -halfZ + (iZ + (stepIZ > 0 ? 1 : 0)) * binSizeZ;
        tMaxZ = tmin + (nextBoundZ - entryZ) / dirZ;
        tDeltaZ = binSizeZ / std::abs(dirZ);
    } else {
        tMaxZ = 1e30;
        tDeltaZ = 1e30;
    }

    // DDA 순회
    G4double tCurrent = tmin;
    G4int maxSteps = fNBinsX + fNBinsY + fNBinsZ + 10;

    for (G4int s = 0; s < maxSteps; s++) {
        if (iX < 0 || iX >= fNBinsX ||
            iY < 0 || iY >= fNBinsY ||
            iZ < 0 || iZ >= fNBinsZ) break;

        G4double tNext = std::min({tMaxX, tMaxY, tMaxZ, tmax});
        G4double segLen = tNext - tCurrent;

        if (segLen > 0) {
            G4int binIndex = fComponent->GetIndex(iX, iY, iZ);
            if (binIndex >= 0) {
                G4double contrib = segLen * photonWeight / binVolume;
                (*fEvtMap)[binIndex] += contrib;
            }
        }

        if (tNext >= tmax) break;
        tCurrent = tNext;

        if (tMaxX <= tMaxY && tMaxX <= tMaxZ) {
            iX += stepIX;
            tMaxX += tDeltaX;
        } else if (tMaxY <= tMaxZ) {
            iY += stepIY;
            tMaxY += tDeltaY;
        } else {
            iZ += stepIZ;
            tMaxZ += tDeltaZ;
        }
    }
}

// Note: Bug A (BeginOfEvent firing) 은 PhysicsModule 측의 KillProcess /
// GenstepCollectorProcess 에서 event-id 변화 감지로 처리. Scorer 의
// BeginOfTrack 훅은 scorer 의 attached volume 안 track 만 fire 되어
// tier3 (proton 이 ImagingSensor 밖) 시나리오에서 트리거 못함.

void TsScoreGPUOpticalPhotonFluence::UserHookForEndOfEvent() {
    if (!fGPUEngine) return;
    // Fix 78측정-2 lean (2026-05-02): hook-wide sScorerMutex / per-event WaitForPendingGPU
    // 모두 제거. dispatch 는 RunGPUDDAAccumulateLocked / SetScoringAABB / ForceRepropagate
    // 등 wrapper 가 m_gpuMutex 짧게 잡고 처리. cmdQueue FIFO 가 propagate / DDA cmdBuf
    // 자동 직렬화 → host sync 불필요. fEvtMap CPU iterate 도 제거 (per-scorer accumulating
    // GPU buffer 가 EndOfRun 에서 1회 read).

    // 첫 호출 시 지오메트리 캐시 및 스코어링 AABB 등록
    bool firstCall = !fGeomCached;
    if (!fGeomCached) {
        CacheComponentGeometry();
    }

    // 2026-05-28 multi-scorer fix: 첫 등록 event 에선 모든 GPU scorer 콜백이 등록될 때까지
    // propagation 을 보류하고, 마지막으로 등록한 scorer 가 1회만 ForceRepropagate 한다.
    // 그래야 단일 propagation 의 transit hit 을 harvest 루프가 등록된 전 scorer 콜백에
    // 재사용(scorer 별 DDA) → 2번째+ scorer 도 데이터 수신. (과거: 첫 scorer 가 먼저
    // 트리거 → 나머지 콜백 미등록 + EndOfEvent/ForceRepropagate 가 beam/genstep 소비 →
    // 후속 scorer 빈 buffer 0.)
    if (firstCall && fScoringVolumeRegistered) {
        bool trigger = false;
        {
            std::lock_guard<std::mutex> _setLock(sScorerMutex);
            sScorerNamesWithRepropagate.insert(GetName());
            if (!sPropagationTriggered &&
                sScorerNamesWithCallback.size() >= sAllScorerNames.size()) {
                sPropagationTriggered = true;
                trigger = true;
            }
        }
        if (trigger) {
            fAABBRegisteredOnce = true;
            fGPUEngine->ForceRepropagate();   // 전 scorer 콜백 등록 완료 후 1회
        }
        // else: 다른 scorer 등록 대기 — 이번엔 propagation 보류 (beam/genstep 보존)
    } else {
        // 후속 event: 정상 per-event 전파 (이미 처리됐으면 idempotent). 이 시점엔 모든
        // scorer 콜백이 등록돼 있어 harvest 루프가 전 scorer DDA.
        fGPUEngine->EndOfEvent();
    }

    // 2026-05-12 fix: harvest callback (RegisterHarvestCallback 등록) 이 EndOfEvent /
    // ForceRepropagate 내부의 HarvestPendingGPUResults 시점에 DDA dispatch.
    // queue FIFO 가 propagation 보다 DDA 가 먼저 실행되도록 보장 → batch N 의
    // m_transitHitBuffer 데이터를 정확히 read. per-event hook 의 추가 DDA 처리는
    // 불필요 (중복 처리 방지). EndOfRun 의 tail rescue 도 마찬가지.
    return;
}
