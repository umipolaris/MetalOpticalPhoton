// Physics Module for gpuoptical
/**
 * TsGPUOpticalPhysicsModule.cc
 * TOPAS Physics Module Extension: Metal GPU 가속 광학 광자 전파
 *
 * 이 모듈은 TOPAS Extension 시스템에 "gpuoptical" 물리 모듈로 등록됨
 * sv:Ph/Default/Modules = 3 "g4em-standard_opt4" "g4optical" "gpuoptical"
 *
 * 동작:
 * 1. ConstructProcess()에서 광학 광자에 GPU Kill 프로세스 등록
 * 2. 하전 입자에 Genstep Collector 프로세스 등록
 * 3. 광학 광자가 생성되면 즉시 kill → CPU 추적 중단
 * 4. 하전 입자 스텝에서 신틸레이션/체렌코프 genstep 수집
 * 5. GPU Metal 엔진이 genstep 기반으로 광자 전파 수행
 */

#include "TsGPUOpticalPhysicsModule.hh"
#include "TsGPUOpticalPhysics.hh"
#include "TsScoreGPUOpticalPhotonCount.hh"
#include "TsScoreGPUOpticalPhotonFluence.hh"
#include "TsScoreGPUOpticalPhotonSurfaceTrackCount.hh"  // SF1 (2026-04-24)
#include "TsScoreGPUOpticalPhotonPhaseSpace.hh"  // 2026-05-19 phsp scorer
#include "TsParameterManager.hh"

#include "G4OpticalPhoton.hh"
#include "G4ProcessManager.hh"
#include "G4ParticleTable.hh"
#include "G4ParticleDefinition.hh"
#include "G4EventManager.hh"
#include "G4Event.hh"
#include "G4RunManager.hh"
#include "G4MTRunManager.hh"
#include "G4Scintillation.hh"
#include "G4Cerenkov.hh"

#include <iostream>
#include <mutex>
#include <algorithm>

// Bug A fix (2026-04-22): event-id 변화 감지 trigger.
// TsGPUOpticalPhysics::BeginOfEvent() 가 어디서도 직접 호출되지 않아
// m_batchEventCount 가 1 에 고정 → FlushBatch firing 안 함 → 모든 primary
// 광자/genstep 이 단일 batch 로 누적되는 버그가 있었음.
//
// Scorer hook 은 attached volume 내 track 만 fire 되어 tier3 (proton 이
// ImagingSensor 밖) 시나리오에서 트리거 못함. 대신 GenstepCollector
// (charged particle) / KillProcess (opticalphoton) 양쪽에서 event-id
// 변화 감지하여 BeginOfEvent 호출. thread_local 로 thread 별 last event id
// 추적.
namespace {
    inline void MaybeTriggerBeginOfEvent(TsGPUOpticalPhysics* engine) {
        if (!engine) return;
        thread_local int lastEventId = -1;
        const G4Event* evt = G4EventManager::GetEventManager()->GetConstCurrentEvent();
        int currentEventId = evt ? evt->GetEventID() : -1;
        if (currentEventId != lastEventId) {
            engine->BeginOfEvent();
            lastEventId = currentEventId;
        }
    }
}

// ============================================================
// GenstepCollector PostStepDoIt 구현
// ============================================================
G4VParticleChange* TsGPUGenstepCollectorProcess::PostStepDoIt(
    const G4Track& track, const G4Step& step) {
    // 2026-05-18 MT race fix (정석): G4VProcess::aParticleChange 는 shared
    // member — 20 thread 가 동시 Initialize() 호출 시 race → G4Cerenkov::
    // thePhysicsTable 0x98 nullptr deref (G4 stepping pipeline state corrupt).
    // G4ThreadLocal per-thread G4ParticleChange 인스턴스 사용 — 정석.
    static G4ThreadLocal G4ParticleChange* tParticleChange = nullptr;
    if (!tParticleChange) tParticleChange = new G4ParticleChange();
    tParticleChange->Initialize(track);

    // Bug A fix: 새 이벤트 첫 charged-particle step 에서 BeginOfEvent 트리거.
    MaybeTriggerBeginOfEvent(fEngine);

    if (fEngine) {
        fEngine->ProcessStep(&step);
    }

    return tParticleChange;
}

// ============================================================
// GPUOpticalKill PostStepDoIt 구현
// validationMode 외에는, primary opticalphoton (parentID==0,
// currentStepNumber==1) 첫 step에서 위치/방향/편광/에너지를 캡처해
// fEngine의 GPU buffer로 push한 후 track을 stop & kill.
// ============================================================
G4VParticleChange* TsGPUOpticalKillProcess::PostStepDoIt(
    const G4Track& track, const G4Step& /*step*/) {
    aParticleChange.Initialize(track);

    // Bug A fix: 새 이벤트 첫 opticalphoton step 에서 BeginOfEvent 트리거.
    // tier3 (charged primary) 는 GenstepCollector 가, U1 (opticalphoton
    // primary) 는 여기가 trigger.
    MaybeTriggerBeginOfEvent(fEngine);

    // Primary opticalphoton 직접 캡처 — Beam source 지원.
    // Validation mode에서도 GPU 측에 같은 광자를 push하여 CPU/GPU
    // 평행 추적 비교를 가능하게 한다 (kill은 validation mode에서는 안 함).
    if (fEngine && track.GetParentID() == 0 &&
        track.GetCurrentStepNumber() == 1) {
        fEngine->CollectPrimaryPhoton(/*step=*/nullptr, &track);

        // 2026-05-18: opticalphoton primary 가 TOPAS Beam path 통해 들어옴.
        // BEAM kernel (Ph/Default/GPUOptical/BeamPhotons) 가 더 권장:
        //   - race-free (architectural — CPU/GPU hand-off path 우회)
        //   - 빠름 (CPU side photon track 생성 안 함)
        //   - GPU pipeline 본질적 일치
        // MT 환경에서 한 번만 warning 출력 (TOPAS Beam path + threads>1).
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true)) {
            G4RunManager* rm = G4RunManager::GetRunManager();
            G4MTRunManager* mtRM = dynamic_cast<G4MTRunManager*>(rm);
            G4int nThreads = (mtRM != nullptr) ? mtRM->GetNumberOfThreads() : 1;
            if (nThreads > 1) {
                std::cout << "\n[TsGPU] WARNING: TOPAS Beam BeamParticle = \"opticalphoton\" "
                          << "+ NumberOfThreads = " << nThreads << " 사용 중.\n"
                          << "  Race fix 적용됐지만 BEAM kernel 권장 (architectural race-free + 빠름):\n"
                          << "    i:Ph/Default/GPUOptical/BeamPhotons = N\n"
                          << "    i:So/Beam/NumberOfHistoriesInRun = 1  # (BEAM mode)\n"
                          << "  + BeamCenter/BeamDir/BeamEnergy 등 GPU side parameter 사용.\n"
                          << "  자세히: docs/opticalphoton_primary_beam_kernel_policy.md\n"
                          << std::endl;
            }
        }
    }

    if (fValidationMode) {
        // ValidationMode: CPU의 g4optical도 동시에 추적
        aParticleChange.ProposeTrackStatus(fAlive);
    } else {
        aParticleChange.ProposeTrackStatus(fStopAndKill);
    }
    return &aParticleChange;
}

// ============================================================
// 싱글톤 GPU 엔진 (MT 환경에서 모든 스레드가 공유)
// ============================================================
static TsGPUOpticalPhysics* g_sharedGPUEngine = nullptr;
static std::mutex g_engineMutex;
static int g_engineRefCount = 0;

// ============================================================
// 생성자/소멸자
// ============================================================
TsGPUOpticalPhysicsModule::TsGPUOpticalPhysicsModule(TsParameterManager* pM)
    : G4VPhysicsConstructor("GPUOpticalPhysics")
    , fPm(pM)
    , fGPUEngine(nullptr)
    , fKillProcess(nullptr)
    , fCollectorProcess(nullptr)
{
}

TsGPUOpticalPhysicsModule::~TsGPUOpticalPhysicsModule() {
    std::lock_guard<std::mutex> lock(g_engineMutex);
    g_engineRefCount--;

    // 마지막 참조 해제 시 정리
    if (g_engineRefCount <= 0 && g_sharedGPUEngine) {
        // 잔여 genstep 처리
        if (g_sharedGPUEngine->IsGPUEnabled()) {
            uint32_t hits = g_sharedGPUEngine->EndOfEvent();
            if (hits > 0) {
                std::cout << "[TsGPU] Final cleanup: " << hits << " hits" << std::endl;
            }

            // GPU 통계 출력
            MOPPropagationStats stats;
            g_sharedGPUEngine->GetStats(&stats);
            std::cout << "[TsGPU] ======== GPU Statistics ========" << std::endl;
            std::cout << "[TsGPU]   Total photons generated: " << stats.totalPhotonsGenerated << std::endl;
            std::cout << "[TsGPU]   Total photons detected:  " << stats.totalPhotonsDetected << std::endl;
            std::cout << "[TsGPU]   Total GPU time:          " << stats.gpuTimeMs << " ms" << std::endl;
            std::cout << "[TsGPU] ================================" << std::endl;
        }

        // Fix H2: use-after-free 방지
        // 스코어러의 static fGPUEngine 포인터가 이 엔진을 참조하고 있음.
        // 엔진 delete 전에 스코어러 쪽 포인터를 먼저 nullptr 로 리셋해
        // Output()이나 남은 훅에서 dangling pointer 접근이 발생하지 않도록 함.
        TsScoreGPUOpticalPhotonCount::SetGPUEngine(nullptr);
        TsScoreGPUOpticalPhotonFluence::SetGPUEngine(nullptr);
        TsScoreGPUOpticalPhotonSurfaceTrackCount::SetGPUEngine(nullptr);  // SF1
        TsScoreGPUOpticalPhotonPhaseSpace::SetGPUEngine(nullptr);  // 2026-05-19 phsp

        delete g_sharedGPUEngine;
        g_sharedGPUEngine = nullptr;
    }
    fGPUEngine = nullptr;
}

// ============================================================
// ConstructParticle - 광학 광자 입자 정의
// ============================================================
void TsGPUOpticalPhysicsModule::ConstructParticle() {
    G4OpticalPhoton::OpticalPhotonDefinition();
}

// ============================================================
// ConstructProcess - GPU 킬 프로세스 + Genstep 수집 프로세스 등록
// ============================================================
void TsGPUOpticalPhysicsModule::ConstructProcess() {
    // 싱글톤 엔진 초기화 (첫 번째 스레드만 실행)
    {
        std::lock_guard<std::mutex> lock(g_engineMutex);
        g_engineRefCount++;

        if (!g_sharedGPUEngine) {

            G4String paramFile = "";
            if (fPm->ParameterExists("Ph/Default/GPUOptical/ParameterFile")) {
                paramFile = fPm->GetStringParameter("Ph/Default/GPUOptical/ParameterFile");
            }

            g_sharedGPUEngine = new TsGPUOpticalPhysics(paramFile);
            g_sharedGPUEngine->SetParameterManager(fPm);
            if (!g_sharedGPUEngine->Initialize()) {
                std::cerr << "[TsGPU] Warning: GPU initialization failed." << std::endl;
                delete g_sharedGPUEngine;
                g_sharedGPUEngine = nullptr;
                return;
            }

            // 배치 임계값 설정
            if (fPm->ParameterExists("Ph/Default/GPUOptical/BatchThreshold")) {
                int threshold = fPm->GetIntegerParameter("Ph/Default/GPUOptical/BatchThreshold");
                g_sharedGPUEngine->SetBatchThreshold((uint32_t)threshold);
            }

            // Fix 63: 이벤트 배치 크기 설정
            if (fPm->ParameterExists("Ph/Default/GPUOptical/EventBatchSize")) {
                int batchSize = fPm->GetIntegerParameter("Ph/Default/GPUOptical/EventBatchSize");
                g_sharedGPUEngine->SetEventBatchSize(batchSize);
            }

            // 로그 레벨 설정
            int logLevel = 1;  // default: geometry register + EndOfRun summary 만. event 별 diag 는 LogLevel>=2 로 set.
            if (fPm->ParameterExists("Ph/Default/GPUOptical/LogLevel")) {
                logLevel = fPm->GetIntegerParameter("Ph/Default/GPUOptical/LogLevel");
            }
            g_sharedGPUEngine->SetLogLevel(logLevel);

            // 2026-05-23: MaxStepsPerPhoton per-run 설정 (default 1,000,000 = CPU TOPAS
            // Ts/MaxStepNumber 맞춤). 영구 TIR 갇힌 광자(예: WLS box 무한흡수 green)가 1M
            // step 까지 bounce 하며 transit hit 폭증 → buffer overflow 하는 셋업은 낮은 값으로 회피.
            if (fPm->ParameterExists("Ph/Default/GPUOptical/MaxStepsPerPhoton")) {
                int maxSteps = fPm->GetIntegerParameter("Ph/Default/GPUOptical/MaxStepsPerPhoton");
                if (maxSteps > 0) {
                    MOPSimConfig cfg = g_sharedGPUEngine->GetConfig();
                    cfg.maxStepsPerPhoton = (uint32_t)maxSteps;
                    g_sharedGPUEngine->SetConfig(cfg);
                    if (g_sharedGPUEngine->GetLogLevel() >= 1)
                        std::cout << "[TsGPU] MaxStepsPerPhoton = " << maxSteps << std::endl;
                }
            }

            // 2026-05-12: TOPAS Ts/Seed 를 GPU RNG base 로 전파 (BEAM mode 등에서
            // seed 별 다른 결과 보장)
            int topasSeed = 1;  // default
            if (fPm->ParameterExists("Ts/Seed")) {
                topasSeed = fPm->GetIntegerParameter("Ts/Seed");
            }
            g_sharedGPUEngine->SetBaseSeed((uint32_t)topasSeed);
            std::cout << "[TsGPU] Base RNG seed set: " << topasSeed
                      << " (Ts/Seed exists=" << fPm->ParameterExists("Ts/Seed") << ")" << std::endl;

            // 2026-05-12: GPU beam source mode (TOPAS Beam 우회).
            // BeamPhotons > 0 시 첫 G4 event 에 BEAM genstep push, N photons GPU 생성.
            // 2026-05-15: TOPAS `i:` 파라미터 max=2^31-1=2.15B → 100B 같은 큰 수 못 받음.
            //   `u:` (unitless double) 로 입력하면 fp64 mantissa 53-bit 까지 정확 (~9e15).
            //   둘 다 지원: 우선 `u:` 시도 → 없으면 `i:` fallback.
            if (fPm->ParameterExists("Ph/Default/GPUOptical/BeamPhotons") ||
                fPm->ParameterExists("Ph/Default/GPUOptical/BeamPhotonsLarge")) {
                // 2026-05-27: BeamPhotons 단일 파라미터로 통일. 선언 타입(i:/u:) 자동 감지:
                //   i: 정수(≤2.15B) / u:,d: fp64(~9e15). 따라서 u:BeamPhotons 하나로 모든 N 처리 가능.
                //   BeamPhotonsLarge 는 deprecated 별칭(하위호환용, 계속 허용).
                int64_t beamN;
                if (fPm->ParameterExists("Ph/Default/GPUOptical/BeamPhotonsLarge")) {
                    beamN = (int64_t)fPm->GetUnitlessParameter("Ph/Default/GPUOptical/BeamPhotonsLarge");
                } else {
                    G4String bpType = fPm->GetTypeOfParameter("Ph/Default/GPUOptical/BeamPhotons");
                    beamN = (bpType == "i")
                          ? (int64_t)fPm->GetIntegerParameter("Ph/Default/GPUOptical/BeamPhotons")
                          : (int64_t)fPm->GetUnitlessParameter("Ph/Default/GPUOptical/BeamPhotons");
                }
                if (beamN > 0) {
                    auto getL = [&](const char* name, double def) -> double {
                        // Length 단위 (G4 internal mm)
                        return fPm->ParameterExists(name) ? fPm->GetDoubleParameter(name, "Length") / CLHEP::mm : def;
                    };
                    auto getE = [&](const char* name, double def) -> double {
                        // Energy 단위 (G4 internal MeV) → eV 변환
                        return fPm->ParameterExists(name) ? fPm->GetDoubleParameter(name, "Energy") / CLHEP::eV : def;
                    };
                    auto getU = [&](const char* name, double def) -> double {
                        return fPm->ParameterExists(name) ? fPm->GetUnitlessParameter(name) : def;
                    };
                    // 위치 (mm), 방향 (unit vector)
                    double cx = getL("Ph/Default/GPUOptical/BeamCenterX", 0.0);
                    double cy = getL("Ph/Default/GPUOptical/BeamCenterY", 0.0);
                    double cz = getL("Ph/Default/GPUOptical/BeamCenterZ", 0.0);
                    double dx = getU("Ph/Default/GPUOptical/BeamDirX", 0.0);
                    double dy = getU("Ph/Default/GPUOptical/BeamDirY", -1.0);
                    double dz = getU("Ph/Default/GPUOptical/BeamDirZ", 0.0);
                    // 에너지 (eV)
                    double energy_eV = getE("Ph/Default/GPUOptical/BeamEnergy", 2.5);
                    // position spread (rectangle half-width, mm)
                    double posCX = getL("Ph/Default/GPUOptical/BeamPosCutoffX", 0.0);
                    double posCY = getL("Ph/Default/GPUOptical/BeamPosCutoffY", 0.0);
                    // angular spread (Gaussian sigma + cutoff, rad)
                    double sigX = getU("Ph/Default/GPUOptical/BeamAngSigmaX", 0.0);
                    double sigY = getU("Ph/Default/GPUOptical/BeamAngSigmaY", 0.0);
                    double cutAX = getU("Ph/Default/GPUOptical/BeamAngCutoffX", 0.0);
                    double cutAY = getU("Ph/Default/GPUOptical/BeamAngCutoffY", 0.0);
                    // 2026-05-17: position cutoff shape + spread (Gaussian sigma)
                    // BeamPositionCutoffShape: "Rectangle" (default) or "Ellipse"
                    int posShape = 0;
                    if (fPm->ParameterExists("Ph/Default/GPUOptical/BeamPositionCutoffShape")) {
                        G4String s = fPm->GetStringParameter("Ph/Default/GPUOptical/BeamPositionCutoffShape");
                        if (s == "Ellipse" || s == "ellipse" || s == "ELLIPSE") posShape = 1;
                    }
                    double posSpreadX = getL("Ph/Default/GPUOptical/BeamPositionSpreadX", 0.0);
                    double posSpreadY = getL("Ph/Default/GPUOptical/BeamPositionSpreadY", 0.0);
                    // Polarization (TOPAS: BeamPolarizationX/Y/Z unit vector).
                    // 셋 다 0 또는 미지정 → random transverse.
                    double polX = getU("Ph/Default/GPUOptical/BeamPolarizationX", 0.0);
                    double polY = getU("Ph/Default/GPUOptical/BeamPolarizationY", 0.0);
                    double polZ = getU("Ph/Default/GPUOptical/BeamPolarizationZ", 0.0);
                    int polMode = (polX*polX + polY*polY + polZ*polZ > 1e-20) ? 1 : 0;
                    // 2026-05-17 추가: BeamPositionDistribution / BeamAngularDistribution
                    // / BeamEnergySpread / BeamTimeSpread 명시적 파라미터.
                    auto distEnum = [&](const char* paramName, int defaultDist) -> int {
                        if (!fPm->ParameterExists(paramName)) return defaultDist;
                        G4String s = fPm->GetStringParameter(paramName);
                        // case-insensitive
                        for (auto& ch : s) ch = (char)std::tolower(ch);
                        if (s == "none") return 0;
                        if (s == "flat") return 1;
                        if (s == "gaussian") return 2;
                        if (s == "isotropic") return 3;  // 4π 등방 (angular 만, BeamAngularDistribution 전용)
                        return defaultDist;
                    };
                    // posDist: spread X 또는 Y > 0 → Gaussian, else Flat (cutoff>0) / None (cutoff=0)
                    int posDistDefault = (posSpreadX > 0 || posSpreadY > 0) ? 2
                                       : ((posCX > 0 || posCY > 0) ? 1 : 0);
                    int posDist = distEnum("Ph/Default/GPUOptical/BeamPositionDistribution", posDistDefault);
                    // angDist: sigma > 0 → Gaussian, else None (cut>0 면 Flat)
                    int angDistDefault = (sigX > 0 || sigY > 0) ? 2
                                       : ((cutAX > 0 || cutAY > 0) ? 1 : 0);
                    int angDist = distEnum("Ph/Default/GPUOptical/BeamAngularDistribution", angDistDefault);
                    // Energy spread (%, CPU 와 동일 convention: spread = % * E / 100)
                    double energySpreadPct = getU("Ph/Default/GPUOptical/BeamEnergySpread", 0.0);
                    double energySpreadEv = energySpreadPct * energy_eV / 100.0;
                    // Time spread (Length/Time 단위 — TOPAS Time = ns internally)
                    double timeSpread = fPm->ParameterExists("Ph/Default/GPUOptical/BeamTimeSpread") ?
                                        fPm->GetDoubleParameter("Ph/Default/GPUOptical/BeamTimeSpread", "Time") / CLHEP::ns : 0.0;
                    double timeCutoff = fPm->ParameterExists("Ph/Default/GPUOptical/BeamTimeCutoff") ?
                                        fPm->GetDoubleParameter("Ph/Default/GPUOptical/BeamTimeCutoff", "Time") / CLHEP::ns : 0.0;
                    // Normalize direction
                    double dmag = std::sqrt(dx*dx + dy*dy + dz*dz);
                    if (dmag > 1e-10) { dx /= dmag; dy /= dmag; dz /= dmag; }
                    g_sharedGPUEngine->SetBeamGenstep(cx, cy, cz, dx, dy, dz,
                                                      energy_eV,
                                                      posCX, posCY,
                                                      sigX, sigY, cutAX, cutAY,
                                                      beamN,
                                                      posShape,
                                                      posSpreadX, posSpreadY,
                                                      polMode,
                                                      polX, polY, polZ,
                                                      posDist, angDist,
                                                      energySpreadEv,
                                                      timeSpread, timeCutoff);

                    // 2026-05-17 Multi-beam — Ph/Default/GPUOptical/Beam{2..32}/BeamPhotons 추가 등록.
                    // 각 named beam 은 자신만의 BeamCenter/Dir/PosCutoff/Energy/... 보유 가능.
                    // 공통 settings (BeamPosShape/Distribution/Polarization/Spectrum 등) 은 SimConfig 의 글로벌.
                    for (int bidx = 2; bidx <= 32; ++bidx) {
                        std::string prefix = "Ph/Default/GPUOptical/Beam" + std::to_string(bidx) + "/";
                        std::string bphotons = prefix + "BeamPhotons";
                        std::string bphotonsLarge = prefix + "BeamPhotonsLarge";
                        if (!fPm->ParameterExists(bphotons.c_str()) &&
                            !fPm->ParameterExists(bphotonsLarge.c_str())) continue;
                        int64_t bN = 0;
                        if (fPm->ParameterExists(bphotonsLarge.c_str())) {
                            bN = (int64_t)fPm->GetUnitlessParameter(bphotonsLarge.c_str());
                        } else {
                            G4String bpT = fPm->GetTypeOfParameter(bphotons.c_str());
                            bN = (bpT == "i")
                               ? (int64_t)fPm->GetIntegerParameter(bphotons.c_str())
                               : (int64_t)fPm->GetUnitlessParameter(bphotons.c_str());
                        }
                        if (bN <= 0) continue;
                        auto getLp = [&](const std::string& n, double def) {
                            return fPm->ParameterExists(n.c_str()) ? fPm->GetDoubleParameter(n.c_str(), "Length") / CLHEP::mm : def;
                        };
                        auto getEp = [&](const std::string& n, double def) {
                            return fPm->ParameterExists(n.c_str()) ? fPm->GetDoubleParameter(n.c_str(), "Energy") / CLHEP::eV : def;
                        };
                        auto getUp = [&](const std::string& n, double def) {
                            return fPm->ParameterExists(n.c_str()) ? fPm->GetUnitlessParameter(n.c_str()) : def;
                        };
                        double bcx = getLp(prefix+"BeamCenterX", 0.0);
                        double bcy = getLp(prefix+"BeamCenterY", 0.0);
                        double bcz = getLp(prefix+"BeamCenterZ", 0.0);
                        double bdx = getUp(prefix+"BeamDirX", 0.0);
                        double bdy = getUp(prefix+"BeamDirY", -1.0);
                        double bdz = getUp(prefix+"BeamDirZ", 0.0);
                        double bEv = getEp(prefix+"BeamEnergy", 2.5);
                        double bpcX = getLp(prefix+"BeamPosCutoffX", 0.0);
                        double bpcY = getLp(prefix+"BeamPosCutoffY", 0.0);
                        double bsX = getUp(prefix+"BeamAngSigmaX", 0.0);
                        double bsY = getUp(prefix+"BeamAngSigmaY", 0.0);
                        double bcaX = getUp(prefix+"BeamAngCutoffX", 0.0);
                        double bcaY = getUp(prefix+"BeamAngCutoffY", 0.0);
                        double bdmag = std::sqrt(bdx*bdx + bdy*bdy + bdz*bdz);
                        if (bdmag > 1e-10) { bdx/=bdmag; bdy/=bdmag; bdz/=bdmag; }
                        g_sharedGPUEngine->AddBeamGenstep(bcx, bcy, bcz, bdx, bdy, bdz,
                                                          bEv, bpcX, bpcY,
                                                          bsX, bsY, bcaX, bcaY, bN);
                        std::cout << "[TsGPU] BEAM#" << bidx << " added: " << bN
                                  << " photons, center=(" << bcx << "," << bcy << "," << bcz
                                  << ")mm dir=(" << bdx << "," << bdy << "," << bdz << ")"
                                  << " E=" << bEv << "eV" << std::endl;
                    }

                    // 2026-05-17 BeamEnergySpectrum (CPU TsVGenerator.cc:110-189).
                    // Discrete: SpectrumValues[] = energies, SpectrumWeights[] = bin weight (sum=1).
                    // Continuous: 같은 layout, sub-bin 선형 interp.
                    if (fPm->ParameterExists("Ph/Default/GPUOptical/BeamEnergySpectrumType")) {
                        G4String specType = fPm->GetStringParameter("Ph/Default/GPUOptical/BeamEnergySpectrumType");
                        for (auto& ch : specType) ch = (char)std::tolower(ch);
                        int specTypeEnum = (specType == "continuous") ? 1 : 0;  // default Discrete
                        int nBins = fPm->GetVectorLength("Ph/Default/GPUOptical/BeamEnergySpectrumValues");
                        int wBins = fPm->GetVectorLength("Ph/Default/GPUOptical/BeamEnergySpectrumWeights");
                        if (nBins > 0 && wBins == nBins && nBins <= 256) {
                            G4double* sEnergies = fPm->GetDoubleVector(
                                "Ph/Default/GPUOptical/BeamEnergySpectrumValues", "Energy");
                            G4double* sWeights = fPm->GetUnitlessVector(
                                "Ph/Default/GPUOptical/BeamEnergySpectrumWeights");
                            // 2026-05-19 fix: piecewise linear PDF 의 trapezoidal weightSums
                            // (TOPAS TsVGenerator.cc:165 와 동일 logic, zero-prepend 없이).
                            // weightSums[i] = ∫_{E[0]}^{E[i]} w(E) dE — Continuous: trapezoidal
                            // Discrete: 단순 cumulative sum.
                            std::vector<float> energies(nBins), cumW(nBins);
                            for (int b = 0; b < nBins; ++b) {
                                energies[b] = (float)(sEnergies[b] / CLHEP::eV);
                            }
                            cumW[0] = (specTypeEnum == 1) ? 0.0f : (float)sWeights[0];
                            for (int b = 1; b < nBins; ++b) {
                                if (specTypeEnum == 1) {
                                    // Continuous: trapezoidal integral
                                    cumW[b] = cumW[b-1] + 0.5f * (float)(sWeights[b] + sWeights[b-1]) * (energies[b] - energies[b-1]);
                                } else {
                                    // Discrete: cumulative sum
                                    cumW[b] = cumW[b-1] + (float)sWeights[b];
                                }
                            }
                            // Discrete 만 normalize ([0,1] — shader: r <= cumW[k]).
                            // Continuous 는 un-normalized 적분 ∫w dE 유지 — shader inverse-CDF
                            // (total=cumW.back(), rUn=r*total, deltaE=rhs/W_lo) 가 un-normalized
                            // 단위를 가정. normalize 시 deltaE 가 [0,1] 로 잘려 max E 가
                            // E_lo+1 (예: 2.65eV) 에서 truncate 되는 bug (2026-05-20 fix).
                            if (specTypeEnum != 1) {
                                float cwBack = cumW.back();
                                if (cwBack > 0.0f) {
                                    for (auto& v : cumW) v /= cwBack;
                                }
                            }
                            // 2026-05-19: shader 의 piecewise linear inverse CDF 위해 raw
                            // weights 도 전달 (PDF slope 계산용).
                            std::vector<float> weights(nBins);
                            for (int b = 0; b < nBins; ++b)
                                weights[b] = (float)sWeights[b];
                            // Engine 에 spectrum 전달 (SimConfig update)
                            MOPSimConfig cfg = g_sharedGPUEngine->GetConfig();
                            cfg.beamSpectrumNumBins = (uint32_t)nBins;
                            cfg.beamSpectrumType    = (uint32_t)specTypeEnum;
                            for (int b = 0; b < nBins; ++b) {
                                cfg.beamSpectrumEnergies[b]   = energies[b];
                                cfg.beamSpectrumCumWeights[b] = cumW[b];
                                cfg.beamSpectrumWeights[b]    = weights[b];
                            }
                            g_sharedGPUEngine->SetConfig(cfg);
                            std::cout << "[TsGPU] BEAM energy spectrum: " << (specTypeEnum ? "Continuous" : "Discrete")
                                      << " " << nBins << " bins, E=[" << energies.front()
                                      << "," << energies.back() << "] eV" << std::endl;
                        } else if (nBins > 256) {
                            std::cerr << "[TsGPU] WARNING: BeamEnergySpectrum " << nBins
                                      << " bins > 256 limit. Spectrum ignored." << std::endl;
                        }
                    }
                    const char* distNames[4] = {"None", "Flat", "Gaussian", "Isotropic"};
                    std::cout << "[TsGPU] BEAM source mode enabled: " << beamN
                              << " photons, center=(" << cx << "," << cy << "," << cz << ")mm"
                              << " dir=(" << dx << "," << dy << "," << dz << ")"
                              << " E=" << energy_eV << "eV ±" << energySpreadEv << "eV"
                              << " posDist=" << distNames[posDist]
                              << " posShape=" << (posShape ? "Ellipse" : "Rectangle")
                              << " posSpread=(" << posSpreadX << "," << posSpreadY << ")mm"
                              << " angDist=" << distNames[angDist]
                              << " pol=" << (polMode ? "fixed" : "random")
                              << "(" << polX << "," << polY << "," << polZ << ")"
                              << " timeSpread=" << timeSpread << "ns"
                              << std::endl;
                }
            }

            // 지오메트리 빌드
            g_sharedGPUEngine->BuildGeometry();

            // GPU 스코어러에 엔진 참조 설정
            TsScoreGPUOpticalPhotonCount::SetGPUEngine(g_sharedGPUEngine);
            TsScoreGPUOpticalPhotonFluence::SetGPUEngine(g_sharedGPUEngine);
            TsScoreGPUOpticalPhotonSurfaceTrackCount::SetGPUEngine(g_sharedGPUEngine);  // SF1
            TsScoreGPUOpticalPhotonPhaseSpace::SetGPUEngine(g_sharedGPUEngine);  // 2026-05-19 phsp

            // Fix 39: MT 지원 완료 — 스레드별 genstep 수집 + GPU 직렬화
            // (경고 메시지 제거)

            if (g_sharedGPUEngine->GetLogLevel() >= 1)
                std::cout << "[TsGPU] GPU Optical Physics Module initialized successfully" << std::endl;
        }
    }

    // 공유 엔진 참조
    fGPUEngine = g_sharedGPUEngine;
    if (!fGPUEngine) return;

    // ValidationMode 확인: GPU와 CPU가 동시에 같은 광자를 추적
    G4bool validationMode = false;
    if (fPm->ParameterExists("Ph/Default/GPUOptical/ValidationMode")) {
        validationMode = fPm->GetBooleanParameter("Ph/Default/GPUOptical/ValidationMode");
        if (validationMode && fGPUEngine->GetLogLevel() >= 1) {
            std::cout << "[TsGPU] *** VALIDATION MODE ENABLED ***" << std::endl;
            std::cout << "[TsGPU]   광학 광자를 kill하지 않음 → CPU g4optical + GPU 동시 추적" << std::endl;
            std::cout << "[TsGPU]   GPU transit hit fluence와 CPU Fluence를 직접 비교 가능" << std::endl;
        }
    }

    // 광학 광자에 GPU Kill 프로세스 등록 (각 스레드에서)
    G4ParticleDefinition* opticalphoton = G4OpticalPhoton::OpticalPhotonDefinition();
    G4ProcessManager* pManager = opticalphoton->GetProcessManager();
    if (pManager) {
        fKillProcess = new TsGPUOpticalKillProcess("GPUOpticalKill", validationMode);
        // Primary opticalphoton (Beam source) 캡처를 위해 엔진 핸들 주입
        fKillProcess->SetEngine(fGPUEngine);
        pManager->AddProcess(fKillProcess, -1, -1, 1);
    }

    // 하전 입자에 Genstep Collector 프로세스 등록 (공유 엔진 사용)
    fCollectorProcess = new TsGPUGenstepCollectorProcess(fGPUEngine, "GPUGenstepCollector");

    auto particleIterator = GetParticleIterator();
    particleIterator->reset();
    int nRegistered = 0;
    int nScintDisabled = 0, nCerenkovDisabled = 0;
    while ((*particleIterator)()) {
        G4ParticleDefinition* particle = particleIterator->value();
        G4double charge = particle->GetPDGCharge();

        if (charge != 0 && particle != opticalphoton) {
            G4ProcessManager* pm = particle->GetProcessManager();
            if (pm) {
                pm->AddProcess(fCollectorProcess, -1, -1, ordDefault);
                nRegistered++;

                // 2026-05-03 perf fix: G4Scintillation 의 photon spawn loop disable.
                // CPU 측에서 매 이벤트 ~44K G4Track allocation (100 MeV proton) 이
                // wall time 의 92% 를 차지. 우리 GPU pipeline 은 ProcessStep 에서
                // 별도 genstep 수집 → GPU 가 photon spawn 하므로 CPU side spawn 은
                // 완전 redundant. SetStackPhotons(false) 로 yield 계산만 하고 secondary
                // 객체 생성/추적 skip → CPU side 대폭 단축 (예상 5-10×).
                // 2026-05-18 디버그 진단: SetStackPhotons(false) 가 G4 ionization
                // 의 secondary 생성 영향 의심 (C12 200 MeV/u GPU mode 의 e- track
                // 4.8% under, 430 MeV/u 37% under measurement). 임시 disable.
                G4ProcessVector* pv = pm->GetProcessList();
                for (G4int ip = 0; ip < (G4int)pv->size(); ip++) {
                    G4VProcess* proc = (*pv)[ip];
                    if (auto* sc = dynamic_cast<G4Scintillation*>(proc)) {
                        // 2026-05-26 재활성: CPU 광자 stacking(8B G4Track 생성/Kill = CPU 시간의
                        // 95% 오버헤드, README 16× 가속의 핵심) 제거. ValidationMode 는 G4 가 광자를
                        // 직접 추적해야 하므로(stacking 끄면 CPU 비교측 빈값) 보존.
                        if (!validationMode) sc->SetStackPhotons(false);
                        nScintDisabled++;
                    }
                    if (auto* ce = dynamic_cast<G4Cerenkov*>(proc)) {
                        if (!validationMode) ce->SetStackPhotons(false);  // 2026-05-26 재활성 (위와 동일, ValidationMode 보존)
                        nCerenkovDisabled++;
                        // 2026-05-18 MT race fix (정석): G4Cerenkov::thePhysicsTable
                        // 가 worker thread 에서 nullptr 인 상태로 PostStepGetPhys 호출되
                        // 면 EXC_BAD_ACCESS at addr 0x98 (`(*thePhysicsTable)[matIdx]`
                        // 의 nullptr deref). G4OpticalPhysics 가 worker thread 의
                        // BuildPhysicsTable 호출을 보장 안 함 → 우리가 explicit 호출.
                        // BuildPhysicsTable 자체는 `if (thePhysicsTable) return;` guard
                        // 가 있어 idempotent 안전 + thread-safe (per-instance member).
                        ce->PreparePhysicsTable(*particle);
                        ce->BuildPhysicsTable(*particle);
                    }
                    if (auto* sc = dynamic_cast<G4Scintillation*>(proc)) {
                        // 같은 이유로 Scintillation 도 explicit pre-build
                        sc->PreparePhysicsTable(*particle);
                        sc->BuildPhysicsTable(*particle);
                    }
                }
            }
        }
    }
    if (fGPUEngine && fGPUEngine->GetLogLevel() >= 1) {
        std::cout << "[TsGPU] StackPhotons disabled: Scintillation=" << nScintDisabled
                  << " Cerenkov=" << nCerenkovDisabled
                  << " | Collector registered=" << nRegistered
                  << " (CPU photon spawn skipped → GPU 직접 처리)" << std::endl;
    }
}
