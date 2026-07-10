// Extra Class for TsGPUOpticalPhysics
/**
 * TsGPUOpticalPhysics.cc
 * TOPAS Extension 구현: GPU 가속 광학 물리 모듈
 *
 * Geant4 SteppingAction에서 신틸레이션/체렌코프 genstep을 수집하고
 * Metal GPU 엔진을 통해 광학 광자를 전파
 */

#include "TsGPUOpticalPhysics.hh"
#include "TsParameterManager.hh"

// Geant4 헤더
#include "G4Material.hh"
#include "G4MaterialPropertiesTable.hh"
#include "G4PhysicalConstants.hh"  // 2026-05-26: Rayleigh MFP (k_Boltzmann, h_Planck, c_light)
#include "G4SystemOfUnits.hh"
#include "G4LogicalVolume.hh"
#include "G4VPhysicalVolume.hh"
#include "G4TransportationManager.hh"
#include "G4Navigator.hh"
#include "G4VSolid.hh"
#include "G4Polyhedron.hh"
#include "G4VProcess.hh"
#include "G4ProcessManager.hh"
#include "G4ParticleDefinition.hh"
#include "G4DynamicParticle.hh"
#include "G4StepPoint.hh"
#include "G4TouchableHistory.hh"
#include "G4AffineTransform.hh"
#include "G4TessellatedSolid.hh"
#include "G4VFacet.hh"
#include "G4Box.hh"  // G1: G4Box analytic intersection 추출
#include "G4Sphere.hh"  // Phase 1 (2026-05-05): G4Sphere analytic intersection
#include "G4Tubs.hh"    // Phase 2 (2026-05-05): G4Tubs/TsCylinder analytic intersection
#include "G4Torus.hh"   // Phase 3 (2026-05-05): G4Torus quartic analytic intersection
#include "G4Polyhedra.hh"          // Phase 4 (2026-05-18): G4Polyhedra (N-sided prism) analytic
#include "G4PolyhedraHistorical.hh"
#include "G4Threading.hh"  // 2026-05-07: GPU 모드는 single-thread 강제 (multi-thread race)
#include "G4RunManager.hh"
#include "G4MTRunManager.hh"

#include "G4LogicalBorderSurface.hh"
#include "G4LogicalSkinSurface.hh"
#include "G4OpticalSurface.hh"
#include "G4NistManager.hh"
#include "G4Poisson.hh"
#include "G4EmSaturation.hh"
#include "G4LossTableManager.hh"
#include "Randomize.hh"

#include <iostream>
#include <cmath>
#include <mutex>
#include <utility>

// Geant4 버전에 무관하게 surface 테이블 원소에서 surface 포인터를 꺼낸다 (#if 불필요):
//   Geant4 <= 11.1 : std::vector<G4Logical{Skin,Border}Surface*>      (원소 = 포인터)
//   Geant4 11.2+   : std::map<key, ...Surface*>                       (원소 = pair)
// 포인터면 그대로, pair 면 .second — overload 해소로 버전별 코드가 자동 선택된다.
template <class T> static inline T* mopSurfacePtr(T* s) { return s; }
template <class K, class T> static inline T* mopSurfacePtr(const std::pair<K, T*>& p) { return p.second; }

// Multi-scorer race fix (2026-05-01): scorer 의 DDA dispatch 를 m_gpuMutex 안에서
// 호출하기 위해 engine extern API 선언.
extern "C" {
    void MOPEngine_RunGPUDDA_WithMaterial(void* engine,
                                           float tx, float ty, float tz,
                                           float fx, float fy, float fz,
                                           uint32_t nx, uint32_t ny, uint32_t nz,
                                           uint32_t scoringMaterialId);
    // 2026-05-12: uint32 → atomic_float (per-voxel cumulative overflow fix)
    const float* MOPEngine_GetDDABinBuffer(void* engine);
    int MOPEngine_RegisterScorerBinBuffer(void* engine, uint32_t totalBins,
                                          uint32_t nBinsE, float eMinEv, float eMaxEv);
    void MOPEngine_SetScorerEnergyFilter(void* engine, int handle, float eLow, float eHigh);
    void MOPEngine_RunGPUDDA_AccumulateExt(void* engine, int scorerHandle,
                                            float tx, float ty, float tz,
                                            float fx, float fy, float fz,
                                            uint32_t nx, uint32_t ny, uint32_t nz,
                                            uint32_t scoringMaterialId,
                                            uint32_t voxelType,
                                            float rMin, float phiStart, float thetaStart);
    void MOPEngine_RunGPUDDA_Accumulate(void* engine, int scorerHandle,
                                         float tx, float ty, float tz,
                                         float fx, float fy, float fz,
                                         uint32_t nx, uint32_t ny, uint32_t nz,
                                         uint32_t scoringMaterialId);
    // 2026-05-12: uint32 → atomic_float (per-voxel cumulative overflow fix)
    const float* MOPEngine_GetScorerBinBuffer(void* engine, int scorerHandle,
                                                  uint32_t* outCount);
}

// ============================================================
// 생성자/소멸자
// ============================================================
TsGPUOpticalPhysics::TsGPUOpticalPhysics(const std::string& topasParameterFile)
    : m_engine(nullptr)
    , m_initialized(false)
    , m_gpuEnabled(true)
    , m_geometryBuilt(false)
    , m_batchThreshold(100000)  // 10만 광자마다 GPU 전파
    , m_logLevel(1)
    , m_parameterFile(topasParameterFile)
{
}

TsGPUOpticalPhysics::~TsGPUOpticalPhysics() {
    // Fix 71: 파이프라이닝 잔여 결과 수확
    if (m_engine) {
        HarvestPendingGPUResults();
    }
    // Fix 63: 잔여 배치 flush
    if (m_engine && !m_batchGensteps.empty()) {
        FlushBatch();
        HarvestPendingGPUResults();
    }
    if (m_engine) {
        MOPEngine_Destroy(m_engine);
        m_engine = nullptr;
    }
}

// ============================================================
// 초기화
// ============================================================
bool TsGPUOpticalPhysics::Initialize() {
    // ============================================================
    // MT (multi-thread) 상태: 해결됨 — thread > 1 허용
    // ============================================================
    //
    // 과거 이슈 (2026-05-07): GPU run 을 NumberOfThreads = 10~30 으로 돌리면
    //   fluence sum 이 CPU 대비 +14~23% over-count (thread=1 은 +0.1% match).
    //   측정 (RotX=60° torus mirror, 50K): 30 threads G/C=1.2316/photons +14%,
    //   1 thread G/C=1.0010/photons 정확. → multi-thread race 가 photon inflate.
    //   당시엔 SINGLE-THREAD hard guard 로 임시 차단.
    //
    // === RESOLVED (2026-05-18): MT race 정석 fix 완료 ===
    //   2곳 fix: (a) AddPrimaryPhoton mutex + m_isFirstEventOfBatch (batch 첫
    //   BeginOfEvent 만 ResetPrimaryPhotons → photon storage overwrite 방지),
    //   (b) G4Cerenkov::aParticleChange / thePhysicsTable shared-member race
    //   (TsGPUOpticalPhysicsModule.cc). GPU dispatch 는 m_gpuMutex 로 직렬화.
    //   검증: cherenkov / U3 opticalphoton 에서 thread>1 bit-identical.
    //   → hard guard 제거, 모든 benchmark thread=20 default
    //   (docs/session_2026_05_18_summary.md).
    //
    // 재확인 (2026-05-25): 07 cherenkov ×1, thread=20 vs thread=1, 5-seed
    //   G/C 1.0061 vs 1.0044 (동일, scatter 내) → race fix 정상 작동 재확인.
    //
    // 안전장치: 새 mechanism 추가로 race 의심 시 thread=1 로 검증.
    //   TOPAS_GPU_DISALLOW_MT=1 환경변수 → thread>1 이면 abort (legacy safe mode).
    //   주의: GPU dispatch 는 m_gpuMutex 직렬화라 thread 늘려도 GPU 속도이득은 작음
    //   (CPU genstep 수집/transport 만 병렬화됨).
    // ============================================================
    // G4Threading::IsMultithreadedApplication() 은 build-time 설정 (TOPAS 는 항상
    // MT build) 이라 NumberOfThreads 와 무관하게 true. 실제 user 설정은
    // G4MTRunManager::GetNumberOfThreads() 로 확인.
    // 2026-05-18: MT race fix 완료 (AddPrimaryPhoton mutex + m_isFirstEventOfBatch).
    // default permissive — thread > 1 허용. 검증된 case (cherenkov, U3 opticalphoton):
    //   GPU dispatch 직렬화 via m_gpuMutex, bit-identical 결과.
    // 새 mechanism 추가 시 race 재발 가능 — 의심 시 thread=1 fallback.
    // TOPAS_GPU_DISALLOW_MT=1 환경변수 시 강제 thread=1 (legacy safe mode).
    {
        G4RunManager* rm = G4RunManager::GetRunManager();
        G4MTRunManager* mtRM = dynamic_cast<G4MTRunManager*>(rm);
        G4int nThreads = (mtRM != nullptr) ? mtRM->GetNumberOfThreads() : 1;
        const char* envDisallow = std::getenv("TOPAS_GPU_DISALLOW_MT");
        bool disallow = (envDisallow && std::string(envDisallow) == "1");
        if (nThreads != 1 && disallow) {
            std::cerr << "[TsGPU] FATAL: TOPAS_GPU_DISALLOW_MT=1 + NumberOfThreads="
                      << nThreads << " → thread=1 강제. config 변경 필요.\n" << std::endl;
            std::abort();
        }
        if (nThreads != 1) {
            std::cout << "[TsGPU] MT mode: NumberOfThreads = " << nThreads
                      << " (G4 worker thread MT, GPU dispatch serialized via m_gpuMutex)"
                      << std::endl;
        }
    }

    // Metal GPU 엔진 생성
    m_engine = MOPEngine_Create();
    if (!m_engine) {
        std::cerr << "[TsGPU] Error: Failed to create Metal GPU engine" << std::endl;
        std::cerr << "[TsGPU] Falling back to CPU optical photon tracking" << std::endl;
        m_gpuEnabled = false;
        return false;
    }

    MOPEngine_SetLogLevel(m_engine, m_logLevel);

    // GPU 사용 가능 여부 확인
    if (!MOPEngine_IsGPUAvailable(m_engine)) {
        std::cerr << "[TsGPU] Warning: Metal GPU not available" << std::endl;
        m_gpuEnabled = false;
        return false;
    }

    // TOPAS 파라미터 파일에서 광학 속성 로드
    if (!m_parameterFile.empty()) {
        int ret = MOPEngine_LoadTOPASParameters(m_engine, m_parameterFile.c_str());
        if (ret != 0) {
            std::cerr << "[TsGPU] Warning: Failed to load TOPAS parameters from "
                      << m_parameterFile << " (error: " << ret << ")" << std::endl;
        } else {
            if (m_logLevel >= 2)
                std::cout << "[TsGPU] TOPAS optical parameters loaded from "
                          << m_parameterFile << std::endl;
        }
    }

    m_initialized = true;
    if (m_logLevel >= 2)
        std::cout << "[TsGPU] GPU Optical Physics module initialized" << std::endl;
    return true;
}

// ============================================================
// TOPAS 파라미터 로드
// ============================================================
int TsGPUOpticalPhysics::LoadTOPASParameters(const std::string& filePath) {
    if (!m_engine) return -1;
    m_parameterFile = filePath;
    return MOPEngine_LoadTOPASParameters(m_engine, filePath.c_str());
}

int TsGPUOpticalPhysics::LoadAdditionalParameters(const std::string& filePath) {
    if (!m_engine) return -1;
    return MOPEngine_LoadTOPASParameters(m_engine, filePath.c_str());
}

// ============================================================
// 지오메트리 구축
// ============================================================
// ============================================================
// G4MaterialPropertyVector → MOPPropertyTable 변환 헬퍼
// ============================================================
static void FillPropertyTableFromG4(MOPPropertyTable& table,
                                     G4MaterialPropertyVector* vec) {
    if (!vec) { table.count = 0; return; }
    size_t n = vec->GetVectorLength();
    if (n > MOP_MAX_PROPERTY_ENTRIES) {
        // 균일 다운샘플링: 첫/끝 포함하여 MAX 개 균일 선택
        table.count = MOP_MAX_PROPERTY_ENTRIES;
        for (size_t j = 0; j < MOP_MAX_PROPERTY_ENTRIES; j++) {
            size_t srcIdx = j * (n - 1) / (MOP_MAX_PROPERTY_ENTRIES - 1);
            table.energies[j] = (float)(vec->Energy(srcIdx) / CLHEP::eV);
            table.values[j]   = (float)((*vec)[srcIdx]);
        }
    } else {
        table.count = (uint32_t)n;
        for (size_t i = 0; i < n; i++) {
            table.energies[i] = (float)(vec->Energy(i) / CLHEP::eV);
            table.values[i]   = (float)((*vec)[i]);
        }
    }
}

// 길이 속성용 (Geant4 내부 → mm)
static void FillLengthPropertyFromG4(MOPPropertyTable& table,
                                      G4MaterialPropertyVector* vec) {
    if (!vec) { table.count = 0; return; }
    size_t n = vec->GetVectorLength();
    if (n > MOP_MAX_PROPERTY_ENTRIES) {
        // 균일 다운샘플링
        table.count = MOP_MAX_PROPERTY_ENTRIES;
        for (size_t j = 0; j < MOP_MAX_PROPERTY_ENTRIES; j++) {
            size_t srcIdx = j * (n - 1) / (MOP_MAX_PROPERTY_ENTRIES - 1);
            table.energies[j] = (float)(vec->Energy(srcIdx) / CLHEP::eV);
            table.values[j]   = (float)((*vec)[srcIdx] / CLHEP::mm);
        }
    } else {
        table.count = (uint32_t)n;
        for (size_t i = 0; i < n; i++) {
            table.energies[i] = (float)(vec->Energy(i) / CLHEP::eV);
            table.values[i]   = (float)((*vec)[i] / CLHEP::mm);
        }
    }
}

void TsGPUOpticalPhysics::BuildGeometry() {
    if (!m_engine || !m_initialized) return;

    // === 1. Geant4 물질 테이블에서 광학 속성 추출 → GPU 등록 ===
    const G4MaterialTable* matTable = G4Material::GetMaterialTable();
    uint32_t registeredMats = 0;

    for (size_t i = 0; i < matTable->size(); i++) {
        G4Material* g4mat = (*matTable)[i];
        G4MaterialPropertiesTable* mpt = g4mat->GetMaterialPropertiesTable();

        MOPMaterialProperties props;
        memset(&props, 0, sizeof(props));
        props.resolutionScale = 1.0f;
        props.yieldRatio = 1.0f;

        if (mpt) {
            props.isOpticalEnabled = 1;

            // 벡터 속성
            FillPropertyTableFromG4(props.refractiveIndex,
                mpt->GetProperty("RINDEX"));
            FillLengthPropertyFromG4(props.absorptionLength,
                mpt->GetProperty("ABSLENGTH"));
            FillPropertyTableFromG4(props.fastComponent,
                mpt->GetProperty("SCINTILLATIONCOMPONENT1"));
            FillPropertyTableFromG4(props.slowComponent,
                mpt->GetProperty("SCINTILLATIONCOMPONENT2"));
            // 레거시 이름도 체크
            if (props.fastComponent.count == 0)
                FillPropertyTableFromG4(props.fastComponent,
                    mpt->GetProperty("FASTCOMPONENT"));
            if (props.slowComponent.count == 0)
                FillPropertyTableFromG4(props.slowComponent,
                    mpt->GetProperty("SLOWCOMPONENT"));

            // M6 수정: 레일리 산란 길이 추출
            FillLengthPropertyFromG4(props.rayleighLength,
                mpt->GetProperty("RAYLEIGH"));

            // 2026-05-26 정석 fix: RAYLEIGH 미지정 시 G4OpRayleigh 와 동일하게 물
            // Rayleigh 산란길이 자동계산 (Einstein-Smoluchowski). G4OpRayleigh.cc
            // CalculateRayleighMeanFreePaths (G4 11.1.3) 와 line-by-line 일치.
            // 누락 시 GPU 가 물 trapped 광자를 G4 보다 과다하게 가둠 — G4 는 Rayleigh
            // 산란이 방향을 임계각 아래로 틀어 일부 탈출시키는데 GPU 는 그 물리를
            // 빠뜨려 over-trap (07 +1% 의 주원인, deeply-trapped 5000cm 에서 +4%).
            // 셰이더 DoRayleighScattering 은 이미 G4 각도분포 재구현 → rayLen 만 채우면 됨.
            if (props.rayleighLength.count == 0) {
                G4MaterialPropertyVector* riVec = mpt->GetProperty("RINDEX");
                G4double betat = 0.0, temperature = 0.0;
                bool doRayleigh = false;
                if (g4mat->GetName() == "Water") {
                    betat = 7.658e-23 * CLHEP::m3 / CLHEP::MeV;  // G4 backwards-compat 상수
                    temperature = 283.15 * CLHEP::kelvin;
                    doRayleigh = true;
                } else if (mpt->ConstPropertyExists("ISOTHERMAL_COMPRESSIBILITY")) {
                    betat = mpt->GetConstProperty("ISOTHERMAL_COMPRESSIBILITY");
                    temperature = g4mat->GetTemperature();
                    doRayleigh = true;
                }
                if (doRayleigh && riVec && riVec->GetVectorLength() >= 2) {
                    G4double scaleFactor = 1.0;
                    if (mpt->ConstPropertyExists("RS_SCALE_FACTOR"))
                        scaleFactor = mpt->GetConstProperty("RS_SCALE_FACTOR");
                    const G4double c1 = scaleFactor * betat * temperature
                                        * CLHEP::k_Boltzmann / (6.0 * CLHEP::pi);
                    size_t nri = riVec->GetVectorLength();
                    size_t cnt = std::min(nri, (size_t)MOP_MAX_PROPERTY_ENTRIES);
                    props.rayleighLength.count = (uint32_t)cnt;
                    for (size_t j = 0; j < cnt; j++) {
                        size_t idx = (cnt < nri) ? (j * (nri - 1) / (cnt - 1)) : j;
                        G4double energy = riVec->Energy(idx);
                        G4double ri  = (*riVec)[idx];
                        G4double ri2 = ri * ri;
                        G4double xlambda = CLHEP::h_Planck * CLHEP::c_light / energy;
                        G4double c2 = std::pow(CLHEP::twopi / xlambda, 4);
                        G4double c3 = std::pow((ri2 - 1.0) * (ri2 + 2.0) / 3.0, 2);
                        G4double mfp = 1.0 / (c1 * c2 * c3);
                        props.rayleighLength.energies[j] = (float)(energy / CLHEP::eV);
                        props.rayleighLength.values[j]   = (float)(mfp / CLHEP::mm);
                    }
                    if (m_logLevel >= 1)
                        std::cout << "[TsGPU] Auto-computed Rayleigh MFP for '"
                                  << g4mat->GetName() << "' (" << cnt
                                  << " pts, G4OpRayleigh Einstein-Smoluchowski)" << std::endl;
                }
            }

            FillLengthPropertyFromG4(props.wlsAbsLength,
                mpt->GetProperty("WLSABSLENGTH"));
            FillPropertyTableFromG4(props.wlsComponent,
                mpt->GetProperty("WLSCOMPONENT"));
            FillLengthPropertyFromG4(props.mieScattering,
                mpt->GetProperty("MIEHG"));

            // 스칼라 속성
            if (mpt->ConstPropertyExists("SCINTILLATIONYIELD"))
                props.scintillationYield = (float)(mpt->GetConstProperty("SCINTILLATIONYIELD")
                                                    * CLHEP::MeV);  // /MeV → /MeV
            if (mpt->ConstPropertyExists("RESOLUTIONSCALE"))
                props.resolutionScale = (float)mpt->GetConstProperty("RESOLUTIONSCALE");
            if (mpt->ConstPropertyExists("SCINTILLATIONTIMECONSTANT1"))
                props.fastTimeConstant = (float)(mpt->GetConstProperty("SCINTILLATIONTIMECONSTANT1")
                                                  / CLHEP::ns);
            else if (mpt->ConstPropertyExists("FASTTIMECONSTANT"))
                props.fastTimeConstant = (float)(mpt->GetConstProperty("FASTTIMECONSTANT")
                                                  / CLHEP::ns);
            if (mpt->ConstPropertyExists("SCINTILLATIONTIMECONSTANT2"))
                props.slowTimeConstant = (float)(mpt->GetConstProperty("SCINTILLATIONTIMECONSTANT2")
                                                  / CLHEP::ns);
            else if (mpt->ConstPropertyExists("SLOWTIMECONSTANT"))
                props.slowTimeConstant = (float)(mpt->GetConstProperty("SLOWTIMECONSTANT")
                                                  / CLHEP::ns);
            if (mpt->ConstPropertyExists("SCINTILLATIONYIELD1"))
                props.yieldRatio = (float)mpt->GetConstProperty("SCINTILLATIONYIELD1");
            else if (mpt->ConstPropertyExists("YIELDRATIO"))
                props.yieldRatio = (float)mpt->GetConstProperty("YIELDRATIO");
            // M1 수정: Birks 상수 추출 (Geant4에서는 물질 자체에 저장)
            props.birksConstant = (float)(g4mat->GetIonisation()->GetBirksConstant()
                                          / (CLHEP::mm / CLHEP::MeV));
            if (mpt->ConstPropertyExists("WLSTIMECONSTANT"))
                props.wlsTimeConstant = (float)(mpt->GetConstProperty("WLSTIMECONSTANT")
                                                 / CLHEP::ns);
            if (mpt->ConstPropertyExists("MIEHG_FORWARD"))
                props.miehgForward = (float)mpt->GetConstProperty("MIEHG_FORWARD");
            if (mpt->ConstPropertyExists("MIEHG_BACKWARD"))
                props.miehgBackward = (float)mpt->GetConstProperty("MIEHG_BACKWARD");
            if (mpt->ConstPropertyExists("MIEHG_FORWARD_RATIO"))
                props.miehgForwardRatio = (float)mpt->GetConstProperty("MIEHG_FORWARD_RATIO");
        }

        // Fix 4: 검출기 물질 자동 감지 — 흡수 길이 < 1mm 이면 검출기로 간주
        props.isDetector = 0;
        if (props.absorptionLength.count > 0) {
            float minAbsLen = 1e20f;
            for (uint32_t k = 0; k < props.absorptionLength.count; k++) {
                if (props.absorptionLength.values[k] < minAbsLen)
                    minAbsLen = props.absorptionLength.values[k];
            }
            if (minAbsLen < 1.0f) {  // < 1mm → 검출기 물질 (G4_Si: 0.01mm)
                props.isDetector = 1;
                if (m_logLevel >= 2)
                    std::cout << "[TsGPU] Detector material detected: '" << g4mat->GetName()
                              << "' (absLen=" << minAbsLen << " mm)" << std::endl;
            }
        }

        // NIST 물질 (G4_Si 등)의 광학 속성이 아직 MPT에 반영되지 않은 경우 보정
        // TOPAS가 NIST 물질의 optical properties를 늦게 첨부하는 문제 우회
        if (!mpt) {
            std::string matName = g4mat->GetName();
            // G4_Si: 실리콘 검출기 — 기본 광학 속성 직접 설정
            if (matName == "G4_Si") {
                props.isOpticalEnabled = 1;
                props.refractiveIndex.count = 2;
                props.refractiveIndex.energies[0] = 1.6f;
                props.refractiveIndex.energies[1] = 4.2f;
                props.refractiveIndex.values[0] = 3.5f;
                props.refractiveIndex.values[1] = 3.5f;
                props.absorptionLength.count = 2;
                props.absorptionLength.energies[0] = 1.6f;
                props.absorptionLength.energies[1] = 4.2f;
                props.absorptionLength.values[0] = 0.01f;  // 0.01 mm
                props.absorptionLength.values[1] = 0.01f;
                props.isDetector = 1;
                if (m_logLevel >= 2)
                    std::cout << "[TsGPU] NIST material '" << matName
                              << "' → detector (hardcoded optical properties)" << std::endl;
            }
        }

        uint32_t matId = MOPEngine_RegisterMaterial(m_engine, &props, g4mat->GetName().c_str());
        m_materialMap[g4mat->GetName()] = matId;
        registeredMats++;

        // 물질 정보 출력 (디버그용)
        if (mpt && m_logLevel >= 2) {
            std::cout << "[TsGPU] Material '" << g4mat->GetName()
                      << "' → GPU id=" << matId
                      << " (rindex=" << props.refractiveIndex.count
                      << ", absLen=" << props.absorptionLength.count
                      << ", rayLen=" << props.rayleighLength.count
                      << ", yield=" << props.scintillationYield
                      << ", det=" << props.isDetector << ")" << std::endl;
        }
    }
    if (m_logLevel >= 2)
        std::cout << "[TsGPU] Registered " << registeredMats << " materials from Geant4" << std::endl;

    // === 2. Geant4 표면 추출 → GPU 등록 ===
    const G4LogicalSkinSurfaceTable* skinTable = G4LogicalSkinSurface::GetSurfaceTable();
    uint32_t registeredSurfs = 0;

    if (skinTable) {
        for (auto&& skinEntry : *skinTable) {
            G4LogicalSkinSurface* skinSurf = mopSurfacePtr(skinEntry);  // vector/map 양쪽 호환
            G4OpticalSurface* optSurf = dynamic_cast<G4OpticalSurface*>(skinSurf->GetSurfaceProperty());
            if (!optSurf) continue;

            MOPSurfaceProperties sprops;
            memset(&sprops, 0, sizeof(sprops));
            // Fix 38: Geant4 G4SurfaceType enum → GPU enum 변환
            // Geant4: dielectric_metal=0, dielectric_dielectric=1
            // GPU:    DIELECTRIC_DIELECTRIC=0, DIELECTRIC_METAL=1
            {
                auto g4type = optSurf->GetType();
                if (g4type == dielectric_dielectric)
                    sprops.type = 0;  // MOP_SURFACE_DIELECTRIC_DIELECTRIC
                else if (g4type == dielectric_metal)
                    sprops.type = 1;  // MOP_SURFACE_DIELECTRIC_METAL
                else
                    sprops.type = 0;  // default: dielectric_dielectric
            }
            sprops.finish = (uint32_t)optSurf->GetFinish();
            sprops.model = (uint32_t)optSurf->GetModel();
            sprops.sigmaAlpha = (float)optSurf->GetSigmaAlpha();

            G4MaterialPropertiesTable* smpt = optSurf->GetMaterialPropertiesTable();
            if (smpt) {
                FillPropertyTableFromG4(sprops.reflectivity, smpt->GetProperty("REFLECTIVITY"));
                FillPropertyTableFromG4(sprops.efficiency, smpt->GetProperty("EFFICIENCY"));
                FillPropertyTableFromG4(sprops.transmittance, smpt->GetProperty("TRANSMITTANCE"));
                FillPropertyTableFromG4(sprops.specularLobe, smpt->GetProperty("SPECULARLOBECONSTANT"));
                FillPropertyTableFromG4(sprops.specularSpike, smpt->GetProperty("SPECULARSPIKECONSTANT"));
                FillPropertyTableFromG4(sprops.backScatter, smpt->GetProperty("BACKSCATTERCONSTANT"));
            }

            std::string surfName = optSurf->GetName();

            // TOPAS extension: b:Su/<name>/ForceReflectivity = "True"
            // Strict reflectivity override mode — surface.reflectivity 를
            // 절대 반사 확률로 사용 (Geant4 default 의 Fresnel-gate 우회).
            if (fPm && fPm->ParameterExists(("Su/" + surfName + "/ForceReflectivity").c_str())) {
                if (fPm->GetBooleanParameter(("Su/" + surfName + "/ForceReflectivity").c_str())) {
                    sprops.forceReflectivity = 1;
                }
            }

            uint32_t surfId = MOPEngine_RegisterSurface(m_engine, &sprops, surfName.c_str());
            m_surfaceMap[surfName] = surfId;
            registeredSurfs++;

            // C6 수정: skin surface를 해당 논리 볼륨에 바인딩 — ConvertGeometryToMesh에서 처리
            m_skinSurfaceLogVols[const_cast<G4LogicalVolume*>(skinSurf->GetLogicalVolume())] = surfId;

            if (m_logLevel >= 3) {
                std::cout << "[TsGPU] Skin surface '" << surfName
                          << "' → GPU id=" << surfId
                          << " (refl=" << sprops.reflectivity.count
                          << ", eff=" << sprops.efficiency.count << ")" << std::endl;
            }
        }
    }

    // Border surfaces
    const G4LogicalBorderSurfaceTable* borderTable = G4LogicalBorderSurface::GetSurfaceTable();
    if (borderTable) {
        for (auto&& borderEntry : *borderTable) {
            G4LogicalBorderSurface* borderSurf = mopSurfacePtr(borderEntry);  // vector/map 양쪽 호환
            G4OpticalSurface* optSurf = dynamic_cast<G4OpticalSurface*>(borderSurf->GetSurfaceProperty());
            if (!optSurf) continue;

            std::string surfName = optSurf->GetName();
            if (m_surfaceMap.find(surfName) == m_surfaceMap.end()) {
                MOPSurfaceProperties sprops;
                memset(&sprops, 0, sizeof(sprops));
                // Fix 38: Geant4 G4SurfaceType enum → GPU enum 변환
                {
                    auto g4type = optSurf->GetType();
                    if (g4type == dielectric_dielectric)
                        sprops.type = 0;  // MOP_SURFACE_DIELECTRIC_DIELECTRIC
                    else if (g4type == dielectric_metal)
                        sprops.type = 1;  // MOP_SURFACE_DIELECTRIC_METAL
                    else
                        sprops.type = 0;  // default
                }
                sprops.finish = (uint32_t)optSurf->GetFinish();
                sprops.model = (uint32_t)optSurf->GetModel();
                sprops.sigmaAlpha = (float)optSurf->GetSigmaAlpha();

                G4MaterialPropertiesTable* smpt = optSurf->GetMaterialPropertiesTable();
                if (smpt) {
                    FillPropertyTableFromG4(sprops.reflectivity, smpt->GetProperty("REFLECTIVITY"));
                    FillPropertyTableFromG4(sprops.efficiency, smpt->GetProperty("EFFICIENCY"));
                    FillPropertyTableFromG4(sprops.transmittance, smpt->GetProperty("TRANSMITTANCE"));
                    FillPropertyTableFromG4(sprops.specularLobe, smpt->GetProperty("SPECULARLOBECONSTANT"));
                    FillPropertyTableFromG4(sprops.specularSpike, smpt->GetProperty("SPECULARSPIKECONSTANT"));
                    FillPropertyTableFromG4(sprops.backScatter, smpt->GetProperty("BACKSCATTERCONSTANT"));
                }

                // TOPAS extension: b:Su/<name>/ForceReflectivity = "True"
                if (fPm && fPm->ParameterExists(("Su/" + surfName + "/ForceReflectivity").c_str())) {
                    if (fPm->GetBooleanParameter(("Su/" + surfName + "/ForceReflectivity").c_str())) {
                        sprops.forceReflectivity = 1;
                    }
                }

                uint32_t surfId = MOPEngine_RegisterSurface(m_engine, &sprops, surfName.c_str());
                m_surfaceMap[surfName] = surfId;
                registeredSurfs++;

                if (m_logLevel >= 3) {
                    std::cout << "[TsGPU] Border surface '" << surfName
                              << "' → GPU id=" << surfId << std::endl;
                }
            }

            // C6 수정: border surface를 물리 볼륨 쌍에 바인딩
            G4VPhysicalVolume* pv1 = const_cast<G4VPhysicalVolume*>(borderSurf->GetVolume1());
            G4VPhysicalVolume* pv2 = const_cast<G4VPhysicalVolume*>(borderSurf->GetVolume2());
            if (pv1 && pv2) {
                m_borderSurfacePVs[std::make_pair(pv1, pv2)] = m_surfaceMap[surfName];
            }
        }
    }

    if (registeredSurfs > 0 && m_logLevel >= 2)
        std::cout << "[TsGPU] Registered " << registeredSurfs << " surfaces from Geant4" << std::endl;

    // === 3. 지오메트리 메시 변환 ===
    ConvertGeometryToMesh();

    // === 4. 가속 구조 빌드 ===
    int ret = MOPEngine_BuildAccelerationStructure(m_engine);
    if (ret != 0) {
        std::cerr << "[TsGPU] Warning: Failed to build GPU acceleration structure" << std::endl;
    }

    m_geometryBuilt = true;
    if (m_logLevel >= 2)
        std::cout << "[TsGPU] GPU geometry built" << std::endl;
}

void TsGPUOpticalPhysics::ConvertGeometryToMesh() {
    /**
     * Geant4 지오메트리를 삼각형 메시로 변환
     * C1 수정: G4VSolid → G4Polyhedron → 월드 좌표 삼각형 리스트
     * C6 수정: skin/border surface를 삼각형 surfaceId에 바인딩
     */
    G4Navigator* navigator =
        G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking();
    G4VPhysicalVolume* worldPV = navigator->GetWorldVolume();
    if (!worldPV) return;

    // Fix: worldSize를 실제 Geant4 월드 크기로 설정 (하드코딩 10000mm → 실제 값)
    {
        G4VSolid* worldSolid = worldPV->GetLogicalVolume()->GetSolid();
        G4ThreeVector pMin, pMax;
        worldSolid->BoundingLimits(pMin, pMax);
        float worldX = (float)((pMax.x() - pMin.x()) / CLHEP::mm);
        float worldY = (float)((pMax.y() - pMin.y()) / CLHEP::mm);
        float worldZ = (float)((pMax.z() - pMin.z()) / CLHEP::mm);
        MOPEngine_SetWorldSize(m_engine, worldX, worldY, worldZ);
        if (m_logLevel >= 2)
            std::cout << "[TsGPU] World size: " << worldX << " x " << worldY
                      << " x " << worldZ << " mm" << std::endl;
    }

    uint32_t volumeId = 0;

    // Fix 81 v2: 모든 볼륨이 처리된 뒤 글로벌 coincidence-aware 필터를
    // 적용해야 하므로, MOPEngine_AddVolumeMesh 호출을 deferred-flush로 변경.
    // (volId, owner-PV-name-for-log, triangles) 를 누적했다가 함수 끝에서
    // 일괄 push.
    struct DeferredVolume {
        uint32_t volumeId;
        std::string ownerName;        // 로그용
        uint32_t matIn;
        uint32_t matOut;
        uint32_t surfId;
        G4VPhysicalVolume* pv;  // 정석 fix (2026-04-26): sibling border surface lookup 용
        std::vector<MOPTriangle> triangles;
    };
    std::vector<DeferredVolume> deferredVolumes;

    // Fix 33: 부모/조부모 정보를 전달하여 coincident face의 matOutside 수정
    // Border surface lookup helper (정석 fix, 2026-04-26):
    // G4LogicalBorderSurface 는 directional (pv1 → pv2) 이지만 BVH triangle 의
    // surfaceId 는 단일 값. 양방향 모두 시도하여 매칭. CPU Geant4 동작은
    // "preStep volume == pv1 AND postStep volume == pv2" 시 적용 — GPU 는
    // triangle 의 두 측 (matIn=current, matOut=adjacent) 에 대해 양방향 모두
    // 검사하면 정확히 동등.
    auto lookupBorderSurfId = [this](G4VPhysicalVolume* a, G4VPhysicalVolume* b) -> uint32_t {
        if (!a || !b) return 0;
        auto it1 = m_borderSurfacePVs.find(std::make_pair(a, b));
        if (it1 != m_borderSurfacePVs.end()) return it1->second;
        auto it2 = m_borderSurfacePVs.find(std::make_pair(b, a));
        if (it2 != m_borderSurfacePVs.end()) return it2->second;
        return 0;
    };

    std::function<void(G4VPhysicalVolume*, G4VPhysicalVolume*, uint32_t, G4AffineTransform, G4VSolid*, G4AffineTransform, uint32_t)> processVolume;
    processVolume = [&](G4VPhysicalVolume* pv, G4VPhysicalVolume* parentPV, uint32_t parentMatId, G4AffineTransform parentTransform,
                        G4VSolid* parentSolid, G4AffineTransform parentSolidWorldTransform,
                        uint32_t grandparentMatId) {
        G4LogicalVolume* lv = pv->GetLogicalVolume();
        G4VSolid* solid = lv->GetSolid();
        G4Material* material = lv->GetMaterial();

        // C1 수정: 로컬→월드 변환 행렬 계산
        G4AffineTransform localTransform(pv->GetRotation(), pv->GetTranslation());
        G4AffineTransform worldTransform = parentTransform * localTransform;

        uint32_t matId = GetMOPMaterialId(material);
        uint32_t currentVolumeId = volumeId;

        // C6 수정: 이 볼륨에 대한 skin surface 확인
        uint32_t skinSurfId = 0;
        auto skinIt = m_skinSurfaceLogVols.find(lv);
        if (skinIt != m_skinSurfaceLogVols.end()) {
            skinSurfId = skinIt->second;
        }

        // C6 수정 + 정석 fix (2026-04-26): 이 볼륨 과 부모 간 border surface 확인.
        // 이전 구현 은 (pv가 pair 어느 쪽이든) 매칭 시 모든 face 에 적용 →
        // directional 정보 손실 + sibling surface 까지 모두 attached. 이제는
        // (currentPV, parentPV) directional pair 로만 lookup → parent face 만
        // border surface 적용. Sibling face 는 Fix 82 단계에서 별도 매핑.
        uint32_t borderSurfId = lookupBorderSurfId(pv, parentPV);

        // 2026-06-01: script-level analytic/mesh 선택. b:Ge/<Component>/GPUForceMesh="True"
        //   → 해당 solid 의 analytic 등록을 건너뛰고 mesh(BVH) 경로 사용. 기존 env
        //   (MOP_SPHERE_ANALYTIC=0 / MOP_CYLINDER_ANALYTIC=0) 와 OR 로 동작하며, torus 는
        //   이 파라미터가 유일한 mesh 토글이다(env 없음). benchmark/10 analytic-vs-mesh
        //   곡면 검증에서 동일 geometry 를 두 경로로 돌리기 위함.
        bool forceMeshThisVol = false;
        if (fPm) {
            std::string fmKey = "Ge/" + std::string(pv->GetName()) + "/GPUForceMesh";
            if (fPm->ParameterExists(fmKey.c_str()) && fPm->GetBooleanParameter(fmKey.c_str()))
                forceMeshThisVol = true;
        }

        // G1 (이전 fix): G4Box analytic intersection 등록. Fast ray-box.
        // 2026-04-25 (5명 합의 architecture, Opticks-aligned): box analytic 폐기.
        //   이유: box analytic 은 materialIdOutside 를 부모 매물로 하드코딩 →
        //         sibling adjacency (Sci/ScRefl 등) 정보 손실. Geant4 G4Navigator
        //         가 동적 조회로 인접 sibling 발견하는 동작과 맞지 않음.
        //   해결: 모든 box 도 mesh 추출 → Fix 82 (sibling adjacency rewriting)
        //         가 triangle 단위로 인접 매물 정정 → 단일 mesh BVH 가 source
        //         of truth (Opticks 패턴).
        //   Perf: 박스 수 적어 BVH 비용 미미 (~5-10 boxes × 12 tris).
        // 향후 box analytic 재활성화 시: per-face materialIdOutside[6] 추가 +
        // Fix 82 의 box 버전 동시 구현 필요.
        G4Box* boxSolid = dynamic_cast<G4Box*>(solid);
        bool boxAnalyticRegistered = false;
        // 2026-05-13: Box analytic 재활성화 — 의도된 비활성화 (2026-04-25, sibling
        // adjacency 보호) 가 sim_light Full scenario 25% gap source. CPU G4Box
        // analytic 와 GPU mesh BVH 처리 차이로 multi-photon path 누적 1.8× over.
        // 활성화 후 Full GPU/CPU 0.758 → 0.950 (sub-5%). minimal/nowrap (no surface)
        // 14-17% under 는 sibling adjacency 손실 (별도 trade-off).
        // algorithm 차이 검증)
#if 1
        if (boxSolid && pv != worldPV && matId != parentMatId) {
            G4ThreeVector ex = worldTransform.TransformAxis(G4ThreeVector(1,0,0));
            G4ThreeVector ey = worldTransform.TransformAxis(G4ThreeVector(0,1,0));
            G4ThreeVector ez = worldTransform.TransformAxis(G4ThreeVector(0,0,1));
            bool axisAligned = (std::abs(ex.x() - 1.0) < 1e-6 && std::abs(ex.y()) < 1e-6 && std::abs(ex.z()) < 1e-6 &&
                                std::abs(ey.y() - 1.0) < 1e-6 && std::abs(ey.x()) < 1e-6 && std::abs(ey.z()) < 1e-6 &&
                                std::abs(ez.z() - 1.0) < 1e-6 && std::abs(ez.x()) < 1e-6 && std::abs(ez.y()) < 1e-6);
            if (axisAligned) {
                G4ThreeVector ctr = worldTransform.TransformPoint(G4ThreeVector(0,0,0));
                MOPBoxGeometry box = {};
                box.HLX = (float)(boxSolid->GetXHalfLength() / CLHEP::mm);
                box.HLY = (float)(boxSolid->GetYHalfLength() / CLHEP::mm);
                box.HLZ = (float)(boxSolid->GetZHalfLength() / CLHEP::mm);
                box.centerX = (float)(ctr.x() / CLHEP::mm);
                box.centerY = (float)(ctr.y() / CLHEP::mm);
                box.centerZ = (float)(ctr.z() / CLHEP::mm);
                box.materialIdInside  = matId;
                box.materialIdOutside = parentMatId;
                box.surfaceId         = borderSurfId > 0 ? borderSurfId : skinSurfId;
                box.volumeId          = currentVolumeId;
                MOPEngine_AddBoxGeometry(m_engine, &box);
                boxAnalyticRegistered = true;
            }
        }
#endif
        (void)boxSolid;  // suppress unused warning if box analytic disabled

        // ============================================================
        // Phase 1 (2026-05-05, Opticks 정석): G4Sphere analytic intersection 등록.
        // 메모리 "정석 fix 원칙" — mesh 24-segment polygon 의 systematic +17% bias
        // 는 hack/scaling 으로 못 풀고, CPU 와 동일한 quadratic 식을 GPU 에서 직접
        // 풀어야 함. MVP 범위: full sphere (RMin=0, DPhi=360, DTheta=180).
        // 다른 parameter 면 mesh 경로 fallback (sphereAnalyticRegistered=false).
        // 환경변수 MOP_SPHERE_ANALYTIC=0 으로 강제 mesh fallback (회귀 방지).
        // ============================================================
        bool sphereAnalyticRegistered = false;
        G4Sphere* sphereSolid = dynamic_cast<G4Sphere*>(solid);
        if (sphereSolid && pv != worldPV && matId != parentMatId) {
            const char* envOpt = std::getenv("MOP_SPHERE_ANALYTIC");
            bool optEnabled = !(envOpt && std::string(envOpt) == "0");
            // Phase 1.1 (2026-05-16): RMin>0 (shell) 지원.
            // Phase 1.2 (2026-05-16): theta range (STheta/DTheta) 지원 — hemisphere.
            //   partial phi (DPhi < 2π) 는 여전히 mesh fallback.
            const G4double rmin   = sphereSolid->GetInnerRadius();
            const G4double rmax   = sphereSolid->GetOuterRadius();
            const G4double sphi   = sphereSolid->GetStartPhiAngle();
            const G4double dphi   = sphereSolid->GetDeltaPhiAngle();
            const G4double stheta = sphereSolid->GetStartThetaAngle();
            const G4double dtheta = sphereSolid->GetDeltaThetaAngle();
            bool isFullPhi   = (std::abs(dphi - CLHEP::twopi) < 1e-9);
            bool isValidRmin = (rmin >= 0.0 && rmin < rmax);
            (void)sphi;  // 현재 미사용 (full phi 만 받음)
            if (optEnabled && !forceMeshThisVol && isFullPhi && isValidRmin) {
                G4ThreeVector ctr = worldTransform.TransformPoint(G4ThreeVector(0,0,0));
                MOPSphereGeometry sph = {};
                sph.centerX     = (float)(ctr.x() / CLHEP::mm);
                sph.centerY     = (float)(ctr.y() / CLHEP::mm);
                sph.centerZ     = (float)(ctr.z() / CLHEP::mm);
                sph.radius      = (float)(rmax / CLHEP::mm);
                sph.innerRadius = (float)(rmin / CLHEP::mm);
                sph.thetaStart  = (float)stheta;
                sph.deltaTheta  = (float)dtheta;
                sph.materialIdInside  = matId;
                sph.materialIdOutside = parentMatId;
                sph.surfaceId         = borderSurfId > 0 ? borderSurfId : skinSurfId;
                sph.volumeId          = currentVolumeId;
                MOPEngine_AddSphereGeometry(m_engine, &sph);
                sphereAnalyticRegistered = true;
                if (m_logLevel >= 1) {
                    std::cout << "[TsGPU-Sphere] '" << pv->GetName()
                              << "' analytic registered: Rmax=" << sph.radius
                              << " Rmin=" << sph.innerRadius
                              << "mm thetaStart=" << sph.thetaStart
                              << " deltaTheta=" << sph.deltaTheta << " rad"
                              << " ctr=(" << sph.centerX << "," << sph.centerY
                              << "," << sph.centerZ << ") matIn=" << sph.materialIdInside
                              << " matOut=" << sph.materialIdOutside << std::endl;
                }
            } else if (m_logLevel >= 1 && sphereSolid) {
                std::cout << "[TsGPU-Sphere] '" << pv->GetName()
                          << "' fallback to mesh: optEnabled=" << optEnabled
                          << " isFullPhi=" << isFullPhi
                          << " isValidRmin=" << isValidRmin
                          << " (RMin=" << rmin << " DPhi=" << dphi
                          << " DTheta=" << dtheta << ")" << std::endl;
            }
        }

        // ============================================================
        // Phase 2 (2026-05-05, Opticks 정석): G4Tubs/TsCylinder analytic intersection.
        // MVP: z-axis aligned, full phi.
        // Phase 2.1 (2026-05-16): RMin>0 (tube/shell) 지원 → innerRadius 전달.
        // 회전 / partial phi 는 여전히 mesh fallback.
        // 환경변수 MOP_CYLINDER_ANALYTIC=0 으로 강제 mesh.
        // ============================================================
        bool cylinderAnalyticRegistered = false;
        G4Tubs* tubsSolid = dynamic_cast<G4Tubs*>(solid);
        // 2026-05-17 정석 fix: SameMaterial cyl (matId == parentMatId) 도 cyl analytic
        // 등록. 이전 default 는 mesh fallback → BarrelShell (Air ↔ Air World) 가 mesh
        // polygon (24 sides per π, arc-polygon 0.34mm mismatch) 으로 처리되어 multi-bounce
        // 7.28% systematic forward bias 의 root cause. 사용자 의도 = 항상 cyl analytic.
        // 2-cyl coincident boundary segfault 는 AddCylinderGeometry 의 auto-shift 가 해결.
        if (tubsSolid && pv != worldPV) {
            const char* envOpt = std::getenv("MOP_CYLINDER_ANALYTIC");
            bool optEnabled = !(envOpt && std::string(envOpt) == "0");
            const G4double rmin = tubsSolid->GetInnerRadius();
            const G4double rmax = tubsSolid->GetOuterRadius();
            const G4double dphi = tubsSolid->GetDeltaPhiAngle();
            const G4double hl   = tubsSolid->GetZHalfLength();
            // Z-axis aligned 검사: world transform 의 axis 가 unit z 인지 확인
            G4ThreeVector ez = worldTransform.TransformAxis(G4ThreeVector(0,0,1));
            bool isZAligned = (std::abs(ez.z() - 1.0) < 1e-6
                            && std::abs(ez.x())     < 1e-6
                            && std::abs(ez.y())     < 1e-6);
            // Phase 2.1: RMin>0 (tube) 도 받아들임 (shader 가 inner radius 처리).
            bool isFullPhi = (std::abs(dphi - CLHEP::twopi) < 1e-9);
            bool isValidRmin = (rmin >= 0.0 && rmin < rmax);
            if (optEnabled && !forceMeshThisVol && isFullPhi && isZAligned && isValidRmin) {
                G4ThreeVector ctr = worldTransform.TransformPoint(G4ThreeVector(0,0,0));
                MOPCylinderGeometry cyl = {};
                cyl.centerX     = (float)(ctr.x() / CLHEP::mm);
                cyl.centerY     = (float)(ctr.y() / CLHEP::mm);
                cyl.centerZ     = (float)(ctr.z() / CLHEP::mm);
                cyl.radius      = (float)(rmax / CLHEP::mm);
                cyl.halfLength  = (float)(hl   / CLHEP::mm);
                cyl.innerRadius = (float)(rmin / CLHEP::mm);  // Phase 2.1: tube shell
                cyl.materialIdInside  = matId;
                cyl.materialIdOutside = parentMatId;
                cyl.surfaceId         = borderSurfId > 0 ? borderSurfId : skinSurfId;
                cyl.volumeId          = currentVolumeId;
                // Fix 33 analytic 등가 (2026-05-14): "구멍" cylinder 판정.
                // matId == grandparentMatId (구멍 안이 조부모 매물) 이고
                // parentMatId != grandparentMatId 이면 — 이 cylinder 는 parent
                // solid 를 관통하는 "구멍" (예: Hole(Air) in Wrap_ZEnd(BlackAbsorber)
                // in World(Air)). +z/-z plane 은 진짜 boundary 가 아니라 grandparent
                // 로 통하는 통로다. mesh BVH 의 Fix 33 (구멍 끝면 triangle 삭제) 와
                // 동일하게, shader 가 endCapOpen 시 Z-slab 체크를 skip → 측면만 hit.
                cyl.endCapOpen = (parentMatId != grandparentMatId &&
                                  matId == grandparentMatId) ? 1u : 0u;
                MOPEngine_AddCylinderGeometry(m_engine, &cyl);
                cylinderAnalyticRegistered = true;
                if (m_logLevel >= 1) {
                    std::cout << "[TsGPU-Cyl] '" << pv->GetName()
                              << "' analytic registered: Rmax=" << cyl.radius
                              << " Rmin=" << cyl.innerRadius
                              << "mm HL=" << cyl.halfLength
                              << "mm ctr=(" << cyl.centerX << "," << cyl.centerY
                              << "," << cyl.centerZ << ")" << std::endl;
                }
            } else if (m_logLevel >= 1 && tubsSolid) {
                std::cout << "[TsGPU-Cyl] '" << pv->GetName()
                          << "' fallback to mesh: optEnabled=" << optEnabled
                          << " isFullPhi=" << isFullPhi
                          << " isZAligned=" << isZAligned
                          << " isValidRmin=" << isValidRmin << std::endl;
            }
        }

        // G4Torus analytic (Phase 3.5 generic axis): worldTransform z-axis 임의 방향 OK.
        // MVP: RMin=0, full phi. Other params → mesh fallback.
        bool torusAnalyticRegistered = false;
        G4Torus* torusSolid = dynamic_cast<G4Torus*>(solid);
        if (torusSolid && pv != worldPV && matId != parentMatId) {
            const G4double rmin = torusSolid->GetRmin();
            const G4double rmax = torusSolid->GetRmax();
            const G4double rTor = torusSolid->GetRtor();
            const G4double dphi = torusSolid->GetDPhi();
            G4ThreeVector axisWorld = worldTransform.TransformAxis(G4ThreeVector(0,0,1));
            axisWorld = axisWorld.unit();
            bool isFullTorus = (rmin < 1e-9 * CLHEP::mm)
                            && (std::abs(dphi - CLHEP::twopi) < 1e-9);
            if (isFullTorus && !forceMeshThisVol) {
                G4ThreeVector ctr = worldTransform.TransformPoint(G4ThreeVector(0,0,0));
                MOPTorusGeometry tor = {};
                tor.centerX = (float)(ctr.x() / CLHEP::mm);
                tor.centerY = (float)(ctr.y() / CLHEP::mm);
                tor.centerZ = (float)(ctr.z() / CLHEP::mm);
                tor.rTor    = (float)(rTor / CLHEP::mm);
                tor.rMax    = (float)(rmax / CLHEP::mm);
                tor.axisX   = (float)axisWorld.x();
                tor.axisY   = (float)axisWorld.y();
                tor.axisZ   = (float)axisWorld.z();
                tor.materialIdInside  = matId;
                tor.materialIdOutside = parentMatId;
                tor.surfaceId         = borderSurfId > 0 ? borderSurfId : skinSurfId;
                tor.volumeId          = currentVolumeId;
                MOPEngine_AddTorusGeometry(m_engine, &tor);
                torusAnalyticRegistered = true;
                if (m_logLevel >= 1) {
                    std::cout << "[TsGPU-Torus] '" << pv->GetName()
                              << "' analytic registered: RTor=" << tor.rTor
                              << "mm RMax=" << tor.rMax
                              << "mm ctr=(" << tor.centerX << "," << tor.centerY
                              << "," << tor.centerZ << ")"
                              << " axis=(" << tor.axisX << "," << tor.axisY
                              << "," << tor.axisZ << ")"
                              << " surfaceId=" << tor.surfaceId
                              << std::endl;
                }
            } else if (m_logLevel >= 1) {
                std::cout << "[TsGPU-Torus] '" << pv->GetName()
                          << "' fallback to mesh: isFullTorus=" << isFullTorus << std::endl;
            }
        }

        // Phase 4 (2026-05-18): G4Polyhedra (N-sided prism) analytic intersection.
        //   범용 지원: numSides ≥ 3, full phi (PhiTotal=2π), Num_z_planes ∈ {2, 4 with apex caps}.
        //   - Num_z_planes=2 + Rmin=0: 단순 prism (R 상수 between Z planes)
        //   - Num_z_planes=4 + R=[0,Rmax,Rmax,0]: prism with apex caps (G4SPolyhedra 의 typical 4-corner closed RZ)
        //   기타 RZ profile (cone-like 점진적 R 변화) 는 mesh fallback.
        bool polyAnalyticRegistered = false;
        G4Polyhedra* polySolid = dynamic_cast<G4Polyhedra*>(solid);
        if (polySolid && pv != worldPV && matId != parentMatId) {
            G4PolyhedraHistorical* orig = polySolid->GetOriginalParameters();
            const G4int numSide = orig ? orig->numSide : 0;
            const G4int nz      = orig ? orig->Num_z_planes : 0;
            const G4double phiStart = orig ? orig->Start_angle : 0.0;
            const G4double phiTot   = orig ? orig->Opening_angle : 0.0;
            const bool isFullPhi = std::abs(phiTot - CLHEP::twopi) < 1e-9;
            // straight prism (Num_z_planes=2, constant R) detection
            G4double zMin = 0, zMax = 0, rOuter = 0;
            bool isStraight = false;
            if (nz == 2 && orig->Rmin[0] < 1e-9 && orig->Rmin[1] < 1e-9
                        && std::abs(orig->Rmax[0] - orig->Rmax[1]) < 1e-9) {
                zMin = std::min(orig->Z_values[0], orig->Z_values[1]);
                zMax = std::max(orig->Z_values[0], orig->Z_values[1]);
                rOuter = orig->Rmax[0];
                isStraight = true;
            } else if (nz == 4 && orig->Rmin[0] < 1e-9 && orig->Rmin[1] < 1e-9
                               && orig->Rmin[2] < 1e-9 && orig->Rmin[3] < 1e-9
                               && orig->Rmax[0] < 1e-9 && orig->Rmax[3] < 1e-9
                               && std::abs(orig->Rmax[1] - orig->Rmax[2]) < 1e-9
                               && std::abs(orig->Z_values[0] - orig->Z_values[1]) < 1e-9
                               && std::abs(orig->Z_values[2] - orig->Z_values[3]) < 1e-9) {
                // G4SPolyhedra typical: Z=[a,a,b,b], R=[0,Rmax,Rmax,0] (apex caps).
                // Lateral body Z ∈ [a, b], R = Rmax. (apex caps = singular points, 광자 hit 무관.)
                zMin = std::min(orig->Z_values[0], orig->Z_values[2]);
                zMax = std::max(orig->Z_values[0], orig->Z_values[2]);
                rOuter = orig->Rmax[1];
                isStraight = true;
            }
            if (numSide >= 3 && isFullPhi && isStraight && rOuter > 0) {
                G4ThreeVector axisWorld = worldTransform.TransformAxis(G4ThreeVector(0,0,1)).unit();
                // local frame e1, e2 도 worldTransform 으로 직접 변환 — PhiStart/face
                // midpoint angle 이 G4 local XY 평면 안에서 정의되므로 (e1,e2,ax) 도
                // G4 의 local (x,y,z) 와 정확히 일치해야 한다 (right-handed).
                G4ThreeVector e1World = worldTransform.TransformAxis(G4ThreeVector(1,0,0)).unit();
                G4ThreeVector e2World = worldTransform.TransformAxis(G4ThreeVector(0,1,0)).unit();
                // center = local (0,0, (zMin+zMax)/2) → world
                G4ThreeVector ctr = worldTransform.TransformPoint(G4ThreeVector(0, 0, 0.5 * (zMin + zMax)));
                MOPPolyhedraGeometry poly = {};
                poly.centerX = (float)(ctr.x() / CLHEP::mm);
                poly.centerY = (float)(ctr.y() / CLHEP::mm);
                poly.centerZ = (float)(ctr.z() / CLHEP::mm);
                poly.axisX   = (float)axisWorld.x();
                poly.axisY   = (float)axisWorld.y();
                poly.axisZ   = (float)axisWorld.z();
                poly.e1X = (float)e1World.x();
                poly.e1Y = (float)e1World.y();
                poly.e1Z = (float)e1World.z();
                poly.e2X = (float)e2World.x();
                poly.e2Y = (float)e2World.y();
                poly.e2Z = (float)e2World.z();
                poly.halfLengthAxis = (float)(0.5 * (zMax - zMin) / CLHEP::mm);
                poly.rMax     = (float)(rOuter / CLHEP::mm);
                poly.rMin     = 0.f;
                poly.phiStart = (float)phiStart;
                poly.phiTotal = (float)phiTot;
                poly.numSides = (uint32_t)numSide;
                poly.materialIdInside  = matId;
                poly.materialIdOutside = parentMatId;
                poly.surfaceId         = borderSurfId > 0 ? borderSurfId : skinSurfId;
                poly.volumeId          = currentVolumeId;
                MOPEngine_AddPolyhedraGeometry(m_engine, &poly);
                polyAnalyticRegistered = true;
                if (m_logLevel >= 1) {
                    std::cout << "[TsGPU-Poly] '" << pv->GetName()
                              << "' analytic registered: N=" << numSide
                              << " R=" << poly.rMax
                              << " halfZ=" << poly.halfLengthAxis
                              << " ctr=(" << poly.centerX << "," << poly.centerY
                              << "," << poly.centerZ << ")"
                              << " axis=(" << poly.axisX << "," << poly.axisY
                              << "," << poly.axisZ << ")"
                              << " surfaceId=" << poly.surfaceId
                              << std::endl;
                }
            } else if (m_logLevel >= 1) {
                std::cout << "[TsGPU-Poly] '" << pv->GetName()
                          << "' fallback to mesh: N=" << numSide
                          << " nz=" << nz
                          << " isFullPhi=" << isFullPhi
                          << " isStraight=" << isStraight << std::endl;
            }
        }

        // Fix 42: G4TessellatedSolid 직접 facet 접근 (STL/CAD 메쉬)
        // CreatePolyhedron()의 vertex winding이 일관되지 않아 법선 방향 오류 발생.
        // G4TessellatedSolid::GetFacet()->GetSurfaceNormal()은 Geant4가 tracking에
        // 사용하는 검증된 outward normal을 제공 → Fix 37 centroid 휴리스틱 불필요.
        G4TessellatedSolid* tessSolid = dynamic_cast<G4TessellatedSolid*>(solid);
        std::vector<MOPTriangle> volumeTriangles;
        bool meshExtracted = false;

        if (tessSolid) {
            int nFacets = tessSolid->GetNumberOfFacets();
            // 법선 회전용: worldTransform의 rotation만 추출
            G4ThreeVector worldOrigin = worldTransform.TransformPoint(G4ThreeVector(0,0,0));

            for (int f = 0; f < nFacets; f++) {
                G4VFacet* facet = tessSolid->GetFacet(f);
                int nVerts = facet->GetNumberOfVertices();
                if (nVerts < 3) continue;

                // Geant4의 검증된 outward normal (로컬 좌표)
                G4ThreeVector localN = facet->GetSurfaceNormal();
                // 월드 좌표로 회전 (translation 제거)
                G4ThreeVector worldN = worldTransform.TransformPoint(localN) - worldOrigin;
                double nLen = worldN.mag();
                if (nLen < 1e-10) continue;
                worldN /= nLen;

                // 꼭짓점을 월드 좌표로 변환
                G4ThreeVector wp0 = worldTransform.TransformPoint(facet->GetVertex(0));
                G4ThreeVector wp1 = worldTransform.TransformPoint(facet->GetVertex(1));
                G4ThreeVector wp2 = worldTransform.TransformPoint(facet->GetVertex(2));

                MOPTriangle tri;
                tri.v0x = wp0.x(); tri.v0y = wp0.y(); tri.v0z = wp0.z();
                tri.v1x = wp1.x(); tri.v1y = wp1.y(); tri.v1z = wp1.z();
                tri.v2x = wp2.x(); tri.v2y = wp2.y(); tri.v2z = wp2.z();
                tri.nx = worldN.x(); tri.ny = worldN.y(); tri.nz = worldN.z();

                tri.volumeId = currentVolumeId;
                tri.materialIdInside = matId;
                tri.materialIdOutside = parentMatId;
                tri.surfaceId = borderSurfId > 0 ? borderSurfId : skinSurfId;

                // Fix 81 v2: trivial-coincident-only 필터는 모든 볼륨이 누적된
                // 후 글로벌 패스에서 처리 (TsGPUOpticalPhysics::ApplyCoincidence
                // Filter). 여기서는 그대로 push.
                volumeTriangles.push_back(tri);

                // 사각형 facet → 두 번째 삼각형 (같은 법선)
                if (nVerts == 4) {
                    G4ThreeVector wp3 = worldTransform.TransformPoint(facet->GetVertex(3));
                    MOPTriangle tri2 = tri;
                    tri2.v1x = wp2.x(); tri2.v1y = wp2.y(); tri2.v1z = wp2.z();
                    tri2.v2x = wp3.x(); tri2.v2y = wp3.y(); tri2.v2z = wp3.z();
                    volumeTriangles.push_back(tri2);
                }
            }

            meshExtracted = true;
            if (m_logLevel >= 1) {
                std::cout << "[TsGPU] Volume '" << pv->GetName()
                          << "': Fix42 direct G4TessellatedSolid extraction, "
                          << nFacets << " facets → "
                          << volumeTriangles.size() << " triangles"
                          << " (authoritative Geant4 normals, no centroid correction needed)"
                          << std::endl;
            }
        }

        // 비-tessellated solid: 기존 CreatePolyhedron 경로 + Fix 37
        // O4 Patch A: G1 analytic 등록된 G4Box는 mesh 추출 skip (double-registration 제거)
        // Phase 1/2/3: analytic sphere/cylinder/torus 도 동일 — mesh 추출 skip
        if (!meshExtracted && !boxAnalyticRegistered && !sphereAnalyticRegistered && !cylinderAnalyticRegistered && !torusAnalyticRegistered && !polyAnalyticRegistered) {
        // 곡면 solid (Sphere/Torus/...) 분해 정밀도 24 → 240 segments.
        // 이유 (2026-05-05): 24 → 96 segments 로 Sphere bias 17% → 5.7% 까지
        // 줄였음 (G4 mesh inscribed polygon 효과). 240 segments 로 추가 수렴 검증.
        // sphere boundary error ∝ (π/N)² → 240/96 → 6.25× 추가 감소 기대.
        const G4int prevSteps = G4Polyhedron::GetNumberOfRotationSteps();
        G4Polyhedron::SetNumberOfRotationSteps(240);
        G4Polyhedron* polyhedron = solid->CreatePolyhedron();
        G4Polyhedron::SetNumberOfRotationSteps(prevSteps);
        if (polyhedron) {
            int nFacets = polyhedron->GetNoFacets();

            for (int f = 1; f <= nFacets; f++) {
                G4int nEdges;
                G4Point3D pts[4];
                G4int edgeFlags[4];

                polyhedron->GetFacet(f, nEdges, pts, edgeFlags);

                if (nEdges >= 3) {
                    G4ThreeVector wp0 = worldTransform.TransformPoint(
                        G4ThreeVector(pts[0].x(), pts[0].y(), pts[0].z()));
                    G4ThreeVector wp1 = worldTransform.TransformPoint(
                        G4ThreeVector(pts[1].x(), pts[1].y(), pts[1].z()));
                    G4ThreeVector wp2 = worldTransform.TransformPoint(
                        G4ThreeVector(pts[2].x(), pts[2].y(), pts[2].z()));

                    MOPTriangle tri;
                    tri.v0x = wp0.x(); tri.v0y = wp0.y(); tri.v0z = wp0.z();
                    tri.v1x = wp1.x(); tri.v1y = wp1.y(); tri.v1z = wp1.z();
                    tri.v2x = wp2.x(); tri.v2y = wp2.y(); tri.v2z = wp2.z();

                    // 법선 계산 (월드 좌표 기반)
                    float ex1 = tri.v1x - tri.v0x, ey1 = tri.v1y - tri.v0y, ez1 = tri.v1z - tri.v0z;
                    float ex2 = tri.v2x - tri.v0x, ey2 = tri.v2y - tri.v0y, ez2 = tri.v2z - tri.v0z;
                    float nx = ey1*ez2 - ez1*ey2;
                    float ny = ez1*ex2 - ex1*ez2;
                    float nz = ex1*ey2 - ey1*ex2;
                    float len = std::sqrt(nx*nx + ny*ny + nz*nz);

                    if (len < 1e-10) {
                        continue;
                    }
                    nx /= len; ny /= len; nz /= len;
                    tri.nx = nx; tri.ny = ny; tri.nz = nz;

                    tri.volumeId = currentVolumeId;
                    tri.materialIdInside = matId;
                    tri.materialIdOutside = parentMatId;
                    tri.surfaceId = borderSurfId > 0 ? borderSurfId : skinSurfId;

                    // Fix 81 v2: trivial-coincident-only 필터는 글로벌 패스
                    // (ApplyCoincidenceFilter)에서 처리. 여기서는 그대로 push.
                    volumeTriangles.push_back(tri);

                    if (nEdges == 4) {
                        G4ThreeVector wp3 = worldTransform.TransformPoint(
                            G4ThreeVector(pts[3].x(), pts[3].y(), pts[3].z()));

                        MOPTriangle tri2 = tri;
                        tri2.v1x = wp2.x(); tri2.v1y = wp2.y(); tri2.v1z = wp2.z();
                        tri2.v2x = wp3.x(); tri2.v2y = wp3.y(); tri2.v2z = wp3.z();

                        ex1 = tri2.v1x - tri2.v0x; ey1 = tri2.v1y - tri2.v0y; ez1 = tri2.v1z - tri2.v0z;
                        ex2 = tri2.v2x - tri2.v0x; ey2 = tri2.v2y - tri2.v0y; ez2 = tri2.v2z - tri2.v0z;
                        nx = ey1*ez2 - ez1*ey2;
                        ny = ez1*ex2 - ex1*ez2;
                        nz = ex1*ey2 - ey1*ex2;
                        len = std::sqrt(nx*nx + ny*ny + nz*nz);
                        if (len < 1e-10) continue;
                        nx /= len; ny /= len; nz /= len;
                        tri2.nx = nx; tri2.ny = ny; tri2.nz = nz;

                        // Fix 81 v2: 글로벌 coincidence 필터 (ApplyCoincidence
                        // Filter)에서 처리, 여기서는 그대로 push.
                        volumeTriangles.push_back(tri2);
                    }
                }
            }

            // Fix 37: 법선 방향 일관성 보정 (비-tessellated solid 전용)
            if (!volumeTriangles.empty()) {
                double vcx = 0, vcy = 0, vcz = 0;
                for (const auto& t : volumeTriangles) {
                    vcx += (t.v0x + t.v1x + t.v2x) / 3.0;
                    vcy += (t.v0y + t.v1y + t.v2y) / 3.0;
                    vcz += (t.v0z + t.v1z + t.v2z) / 3.0;
                }
                vcx /= volumeTriangles.size();
                vcy /= volumeTriangles.size();
                vcz /= volumeTriangles.size();

                int flippedCount = 0;
                for (auto& t : volumeTriangles) {
                    float tcx = (t.v0x + t.v1x + t.v2x) / 3.0f - (float)vcx;
                    float tcy = (t.v0y + t.v1y + t.v2y) / 3.0f - (float)vcy;
                    float tcz = (t.v0z + t.v1z + t.v2z) / 3.0f - (float)vcz;
                    float dotVal = t.nx * tcx + t.ny * tcy + t.nz * tcz;
                    if (dotVal < 0.0f) {
                        t.nx = -t.nx;
                        t.ny = -t.ny;
                        t.nz = -t.nz;
                        flippedCount++;
                    }
                }
                if (flippedCount > 0 && m_logLevel >= 2) {
                    std::cout << "[TsGPU] Volume '" << pv->GetName()
                              << "': Fix37 flipped " << flippedCount
                              << " / " << volumeTriangles.size()
                              << " inward-pointing normals to outward"
                              << " (centroid=(" << vcx << "," << vcy << "," << vcz << "))"
                              << std::endl;
                }
            }

            meshExtracted = true;
            delete polyhedron;
        }
        } // end !meshExtracted

        if (meshExtracted) {

            // Fix 29+30: 부모-자식 볼륨 겹침 삼각형 제거 (Coincident Face Problem)
            // Geant4 CSG에서는 자식 볼륨이 부모 영역을 대체하지만,
            // CreatePolyhedron()은 부모 전체 면을 생성함.
            // GPU flat acceleration structure에서 동일 위치 삼각형은 비결정적 →
            // 부모 면이 선택되면 잘못된 경계 처리 발생.
            // 해결: 부모 삼각형을 세분화한 후 자식 볼륨 내부 sub-triangle 제거.
            int nDaughters = lv->GetNoDaughters();
            if (nDaughters > 0 && !volumeTriangles.empty()) {
                size_t originalCount = volumeTriangles.size();

                // 자식 볼륨별 월드→로컬 역변환을 미리 계산
                struct DaughterInfo {
                    G4VSolid* solid;
                    G4AffineTransform inverseTf;
                };
                std::vector<DaughterInfo> daughters(nDaughters);
                for (int d = 0; d < nDaughters; d++) {
                    G4VPhysicalVolume* daughter = lv->GetDaughter(d);
                    G4LogicalVolume* dlv = daughter->GetLogicalVolume();
                    daughters[d].solid = dlv->GetSolid();
                    G4AffineTransform daughterLocalTf(daughter->GetRotation(),
                                                      daughter->GetTranslation());
                    G4AffineTransform daughterWorldTf = worldTransform;
                    daughterWorldTf *= daughterLocalTf;
                    daughters[d].inverseTf = daughterWorldTf.Inverse();
                }

                // 중심점이 자식 내부인지 검사하는 람다
                auto isCentroidInsideChild = [&](G4double cx, G4double cy, G4double cz) -> bool {
                    G4ThreeVector centroidWorld(cx, cy, cz);
                    for (int d = 0; d < nDaughters; d++) {
                        G4ThreeVector centroidLocal = daughters[d].inverseTf.TransformPoint(centroidWorld);
                        EInside pos = daughters[d].solid->Inside(centroidLocal);
                        if (pos == kInside || pos == kSurface) return true;
                    }
                    return false;
                };

                // 삼각형의 꼭짓점 중 하나라도 자식 내부인지 검사 (세분화 필요 여부 판단)
                auto anyVertexInsideChild = [&](const MOPTriangle& tri) -> bool {
                    if (isCentroidInsideChild(tri.v0x, tri.v0y, tri.v0z)) return true;
                    if (isCentroidInsideChild(tri.v1x, tri.v1y, tri.v1z)) return true;
                    if (isCentroidInsideChild(tri.v2x, tri.v2y, tri.v2z)) return true;
                    return false;
                };

                // Fix 30: 삼각형 세분화 후 필터링
                // 큰 삼각형(자식 볼륨보다 클 때)은 중심점 검사만으로 부족.
                // 세분화: 각 삼각형을 N×N 격자로 분할 (바리센트릭 좌표 기반)
                const int SUBDIV_N = 1;  // SUBDIV 무영향 (24 tri 동일)

                std::vector<MOPTriangle> filteredTriangles;
                filteredTriangles.reserve(originalCount);
                size_t totalSubRemoved = 0;
                size_t totalSubCreated = 0;

                for (const auto& tri : volumeTriangles) {
                    // 1단계: 중심점으로 빠른 검사 (완전히 자식 내부)
                    G4double cx = (tri.v0x + tri.v1x + tri.v2x) / 3.0;
                    G4double cy = (tri.v0y + tri.v1y + tri.v2y) / 3.0;
                    G4double cz = (tri.v0z + tri.v1z + tri.v2z) / 3.0;

                    bool centroidInside = isCentroidInsideChild(cx, cy, cz);

                    if (centroidInside && !anyVertexInsideChild(tri)) {
                        // 중심점은 내부이지만 꼭짓점은 외부 → 부분 겹침, 세분화 필요
                        // (이 경우는 드묾, 안전하게 세분화)
                    }

                    if (centroidInside) {
                        // 전체 삼각형이 자식 내부에 있을 가능성 높음
                        // 꼭짓점도 검사: 모든 꼭짓점이 내부이면 전체 제거
                        bool allInside = isCentroidInsideChild(tri.v0x, tri.v0y, tri.v0z)
                                      && isCentroidInsideChild(tri.v1x, tri.v1y, tri.v1z)
                                      && isCentroidInsideChild(tri.v2x, tri.v2y, tri.v2z);
                        if (allInside) {
                            totalSubRemoved++;
                            continue;  // 전체 제거
                        }
                    }

                    // 2단계: 자식 볼륨과의 겹침 가능성 판단 → 세분화 여부
                    // 삼각형 내부의 그리드 포인트를 샘플링하여 자식 볼륨 내부 점이 있는지 검사
                    // (큰 삼각형이 작은 자식 구멍을 완전히 덮는 경우 대응)
                    bool hasOverlap = false;
                    if (!centroidInside) {
                        // 삼각형 내부에 5×5 격자점 샘플링
                        const int PROBE_N = 5;
                        float probeInvN = 1.0f / PROBE_N;
                        for (int pi = 0; pi <= PROBE_N && !hasOverlap; pi++) {
                            for (int pj = 0; pj <= PROBE_N - pi && !hasOverlap; pj++) {
                                float pa = (PROBE_N - pi - pj) * probeInvN;
                                float pb = pi * probeInvN;
                                float pc = pj * probeInvN;
                                float px = pa * tri.v0x + pb * tri.v1x + pc * tri.v2x;
                                float py = pa * tri.v0y + pb * tri.v1y + pc * tri.v2y;
                                float pz = pa * tri.v0z + pb * tri.v1z + pc * tri.v2z;
                                if (isCentroidInsideChild(px, py, pz)) {
                                    hasOverlap = true;
                                }
                            }
                        }
                    }

                    bool needsSubdiv = anyVertexInsideChild(tri) ||
                                       centroidInside ||
                                       hasOverlap;

                    if (!needsSubdiv) {
                        // 겹침 없음 → 원본 유지
                        filteredTriangles.push_back(tri);
                        continue;
                    }

                    // 세분화: 바리센트릭 좌표로 N×N 격자
                    // 각 (i,j) 셀에서 i+j < N인 부분에 2개 삼각형 생성
                    float v0x = tri.v0x, v0y = tri.v0y, v0z = tri.v0z;
                    float v1x = tri.v1x, v1y = tri.v1y, v1z = tri.v1z;
                    float v2x = tri.v2x, v2y = tri.v2y, v2z = tri.v2z;

                    auto interpX = [&](float a, float b, float c) { return a * v0x + b * v1x + c * v2x; };
                    auto interpY = [&](float a, float b, float c) { return a * v0y + b * v1y + c * v2y; };
                    auto interpZ = [&](float a, float b, float c) { return a * v0z + b * v1z + c * v2z; };

                    float invN = 1.0f / SUBDIV_N;

                    for (int i = 0; i < SUBDIV_N; i++) {
                        for (int j = 0; j < SUBDIV_N - i; j++) {
                            // 하삼각형: (i,j), (i+1,j), (i,j+1)
                            float a0 = (SUBDIV_N - i - j) * invN;
                            float b0 = i * invN;
                            float c0 = j * invN;

                            float a1 = (SUBDIV_N - i - 1 - j) * invN;
                            float b1 = (i + 1) * invN;
                            float c1 = j * invN;

                            float a2 = (SUBDIV_N - i - j - 1) * invN;
                            float b2 = i * invN;
                            float c2 = (j + 1) * invN;

                            float sx0 = interpX(a0, b0, c0), sy0 = interpY(a0, b0, c0), sz0 = interpZ(a0, b0, c0);
                            float sx1 = interpX(a1, b1, c1), sy1 = interpY(a1, b1, c1), sz1 = interpZ(a1, b1, c1);
                            float sx2 = interpX(a2, b2, c2), sy2 = interpY(a2, b2, c2), sz2 = interpZ(a2, b2, c2);

                            // sub-triangle 중심점 검사
                            float scx = (sx0 + sx1 + sx2) / 3.0f;
                            float scy = (sy0 + sy1 + sy2) / 3.0f;
                            float scz = (sz0 + sz1 + sz2) / 3.0f;

                            totalSubCreated++;
                            if (isCentroidInsideChild(scx, scy, scz)) {
                                totalSubRemoved++;
                            } else {
                                MOPTriangle sub = tri;  // 원본 속성 복사 (matId, surfId 등)
                                sub.v0x = sx0; sub.v0y = sy0; sub.v0z = sz0;
                                sub.v1x = sx1; sub.v1y = sy1; sub.v1z = sz1;
                                sub.v2x = sx2; sub.v2y = sy2; sub.v2z = sz2;
                                // 법선 재계산
                                float ex1s = sx1-sx0, ey1s = sy1-sy0, ez1s = sz1-sz0;
                                float ex2s = sx2-sx0, ey2s = sy2-sy0, ez2s = sz2-sz0;
                                float snx = ey1s*ez2s - ez1s*ey2s;
                                float sny = ez1s*ex2s - ex1s*ez2s;
                                float snz = ex1s*ey2s - ey1s*ex2s;
                                float slen = std::sqrt(snx*snx + sny*sny + snz*snz);
                                if (slen > 1e-10) { snx/=slen; sny/=slen; snz/=slen; }
                                // 법선 방향이 원본과 같은 방향 유지
                                if (snx*tri.nx + sny*tri.ny + snz*tri.nz < 0) {
                                    snx = -snx; sny = -sny; snz = -snz;
                                }
                                sub.nx = snx; sub.ny = sny; sub.nz = snz;
                                filteredTriangles.push_back(sub);
                            }

                            // 상삼각형: (i+1,j), (i+1,j+1), (i,j+1) — i+j+1 < N일 때만
                            if (i + j + 1 < SUBDIV_N) {
                                float a3 = (SUBDIV_N - i - 1 - j) * invN;
                                float b3 = (i + 1) * invN;
                                float c3 = j * invN;

                                float a4 = (SUBDIV_N - i - 2 - j) * invN;
                                float b4 = (i + 1) * invN;
                                float c4 = (j + 1) * invN;

                                float a5 = (SUBDIV_N - i - j - 1) * invN;
                                float b5 = i * invN;
                                float c5 = (j + 1) * invN;

                                float sx3 = interpX(a3, b3, c3), sy3 = interpY(a3, b3, c3), sz3 = interpZ(a3, b3, c3);
                                float sx4 = interpX(a4, b4, c4), sy4 = interpY(a4, b4, c4), sz4 = interpZ(a4, b4, c4);
                                float sx5 = interpX(a5, b5, c5), sy5 = interpY(a5, b5, c5), sz5 = interpZ(a5, b5, c5);

                                float scx2 = (sx3 + sx4 + sx5) / 3.0f;
                                float scy2 = (sy3 + sy4 + sy5) / 3.0f;
                                float scz2 = (sz3 + sz4 + sz5) / 3.0f;

                                totalSubCreated++;
                                if (isCentroidInsideChild(scx2, scy2, scz2)) {
                                    totalSubRemoved++;
                                } else {
                                    MOPTriangle sub = tri;
                                    sub.v0x = sx3; sub.v0y = sy3; sub.v0z = sz3;
                                    sub.v1x = sx4; sub.v1y = sy4; sub.v1z = sz4;
                                    sub.v2x = sx5; sub.v2y = sy5; sub.v2z = sz5;
                                    float ex1s = sx4-sx3, ey1s = sy4-sy3, ez1s = sz4-sz3;
                                    float ex2s = sx5-sx3, ey2s = sy5-sy3, ez2s = sz5-sz3;
                                    float snx = ey1s*ez2s - ez1s*ey2s;
                                    float sny = ez1s*ex2s - ex1s*ez2s;
                                    float snz = ex1s*ey2s - ey1s*ex2s;
                                    float slen = std::sqrt(snx*snx + sny*sny + snz*snz);
                                    if (slen > 1e-10) { snx/=slen; sny/=slen; snz/=slen; }
                                    if (snx*tri.nx + sny*tri.ny + snz*tri.nz < 0) {
                                        snx = -snx; sny = -sny; snz = -snz;
                                    }
                                    sub.nx = snx; sub.ny = sny; sub.nz = snz;
                                    filteredTriangles.push_back(sub);
                                }
                            }
                        }
                    }
                }

                size_t finalCount = filteredTriangles.size();
                if (m_logLevel >= 2 && (totalSubRemoved > 0 || finalCount != originalCount)) {
                    std::cout << "[TsGPU] Volume '" << pv->GetName()
                              << "': coincident face filter: "
                              << originalCount << " original tris → "
                              << totalSubCreated << " sub-tris created, "
                              << totalSubRemoved << " removed → "
                              << finalCount << " final tris"
                              << std::endl;
                }
                volumeTriangles = std::move(filteredTriangles);
            }

            // Fix 33: 자식 볼륨의 삼각형 중 부모 solid의 면과 coincident한 것 제거
            // GPU ray tracing에서 자식의 끝면이 부모의 끝면과 같은 Z에 있으면
            // matOutside가 부모 물질로 잘못 설정됨 (실제로는 부모 외부의 물질이어야 함).
            // 해결: 자식 삼각형의 모든 꼭짓점이 부모 solid 면 위(kSurface)이면 제거.
            // Fix 33: 자식 볼륨의 "구멍" 삼각형 처리
            // 부모의 외면과 coincident하고, 자식 물질이 조부모 물질과 같으면
            // (= "구멍" 역할) → 해당 삼각형 삭제.
            // GPU에서는 이 면이 없어야 광자가 구멍을 통해 직접 다음 볼륨으로 진행.
            // 조건: matId == grandparentMatId (구멍), parentMatId != grandparentMatId
            // 예: Hole(Air) inside Wrap_ZEnd(BlackAbsorber) in World(Air)
            //     → Hole의 끝면 삭제, 측면은 유지
            // SensorSilicon(Si) inside ImagingSensor(Air) in World(Air)
            //     → matId(Si) != grandparentMatId(Air) → 삭제하지 않음
            if (parentSolid && !volumeTriangles.empty() &&
                parentMatId != grandparentMatId && matId == grandparentMatId) {
                G4AffineTransform parentInverse = parentSolidWorldTransform.Inverse();
                G4ThreeVector pMin, pMax;
                parentSolid->BoundingLimits(pMin, pMax);
                const G4double tol = 0.01 * CLHEP::mm;

                size_t beforeCount = volumeTriangles.size();
                std::vector<MOPTriangle> kept;
                kept.reserve(beforeCount);

                for (const auto& tri : volumeTriangles) {
                    G4ThreeVector v0l = parentInverse.TransformPoint(G4ThreeVector(tri.v0x, tri.v0y, tri.v0z));
                    G4ThreeVector v1l = parentInverse.TransformPoint(G4ThreeVector(tri.v1x, tri.v1y, tri.v1z));
                    G4ThreeVector v2l = parentInverse.TransformPoint(G4ThreeVector(tri.v2x, tri.v2y, tri.v2z));

                    auto near = [](double a, double b, double t) { return std::abs(a - b) < t; };
                    bool sameFace = false;
                    sameFace |= near(v0l.x(), pMin.x(), tol) && near(v1l.x(), pMin.x(), tol) && near(v2l.x(), pMin.x(), tol);
                    sameFace |= near(v0l.x(), pMax.x(), tol) && near(v1l.x(), pMax.x(), tol) && near(v2l.x(), pMax.x(), tol);
                    sameFace |= near(v0l.y(), pMin.y(), tol) && near(v1l.y(), pMin.y(), tol) && near(v2l.y(), pMin.y(), tol);
                    sameFace |= near(v0l.y(), pMax.y(), tol) && near(v1l.y(), pMax.y(), tol) && near(v2l.y(), pMax.y(), tol);
                    sameFace |= near(v0l.z(), pMin.z(), tol) && near(v1l.z(), pMin.z(), tol) && near(v2l.z(), pMin.z(), tol);
                    sameFace |= near(v0l.z(), pMax.z(), tol) && near(v1l.z(), pMax.z(), tol) && near(v2l.z(), pMax.z(), tol);

                    if (!sameFace) {
                        kept.push_back(tri);
                    }
                }

                size_t removedCount = beforeCount - kept.size();
                if (removedCount > 0 && m_logLevel >= 2) {
                    std::cout << "[TsGPU] Volume '" << pv->GetName()
                              << "': removed " << removedCount
                              << " hole-face tris (matId=" << matId
                              << "==grandparentMatId, coincident with parent surface)"
                              << " (" << beforeCount << " → " << kept.size() << ")"
                              << std::endl;
                }
                volumeTriangles = std::move(kept);
            }

            if (!volumeTriangles.empty() && pv != worldPV) {
                // Fix 81 v2: deferred-flush — engine push 대신 누적
                // Fix (2026-04-22, expert 4 분석): worldPV 의 외곽 triangle 은
                // 항상 trivial (matIn=matOut=parentMatId=0) 이고 OUT_OF_WORLD
                // 체크가 정확히 같은 역할을 하므로 BVH 등록 불필요. 등록 시
                // World 외곽 면이 phantom hit 으로 잡혀 reflected 광자가
                // World 안으로 다시 들어오는 multi-bounce path 누적이 GPU/CPU
                // Air fluence 22% 편차의 원인 (특히 -X / z 위쪽 영역).
                DeferredVolume dv;
                dv.volumeId = currentVolumeId;
                dv.ownerName = pv->GetName();
                dv.matIn = matId;
                dv.matOut = parentMatId;
                dv.surfId = (borderSurfId > 0 ? borderSurfId : skinSurfId);
                dv.pv = pv;  // 정석 fix (2026-04-26): sibling border surface lookup 용
                dv.triangles = std::move(volumeTriangles);
                deferredVolumes.push_back(std::move(dv));
            }

        } // end if (meshExtracted)

        // m_volumeMap: volumeId → 물리볼륨 매핑 (스코어링용)
        m_volumeMap[currentVolumeId] = pv;
        volumeId++;

        // 자식 볼륨 재귀 처리 (parentPV = pv)
        for (size_t i = 0; i < (size_t)lv->GetNoDaughters(); i++) {
            processVolume(lv->GetDaughter(i), pv, matId, worldTransform,
                          solid, worldTransform, parentMatId);
        }
    };

    // 월드 볼륨은 항등 변환에서 시작 (parentPV = nullptr, parentSolid = nullptr, grandparentMatId = 0)
    processVolume(worldPV, nullptr, 0, G4AffineTransform(), nullptr, G4AffineTransform(), 0);

    // ============================================================
    // Fix 81 v2: coincidence-aware trivial-boundary filter (글로벌 패스).
    //
    // 배경:
    //   matIn==matOut && surfaceId==0 인 삼각형 ("trivial face") 은 광학적으로
    //   아무 작용도 하지 않지만, BVH ray query 시 일반 hit으로 잡혀 boundary
    //   processing이 'phantom hit'을 consume함. 다음 ray가 tmin 너머의 진짜
    //   boundary를 놓칠 수 있어 U1 multi-volume (Air slab + Glass slab 공면)
    //   같은 케이스에서 굴절이 약화되는 버그가 발생.
    //
    // 단, 모든 trivial face를 drop하면 안 됨:
    //   ImagingSensor (Air-in-Air, parent World/Air) 같이 boundary-hit 기반
    //   scoring을 위해 BVH에 있어야 하는 정상 trivial face가 함께 사라지면서
    //   compound lens scoring이 0이 되는 회귀 발생 (Fix 81 v1 폐기 사유).
    //
    // 해법:
    //   trivial face가 *어떤 non-trivial face와 공면(centroid<10µm, |n·n'|>
    //   0.999)일 때만* drop. 공면 짝이 없는 trivial face (ImagingSensor 등)
    //   는 보존.
    //
    // Edge case (알려진 한계):
    //   부분 overlap (예: 작은 sensor가 큰 슬랩의 일부 면에만 닿음) 에서는
    //   centroid 거리 차이로 공면 검출이 실패하여 phantom hit 버그가 잔존
    //   가능. 사용자 시나리오에서는 드물어 미해결로 둠.
    // ============================================================
    if (!deferredVolumes.empty()) {
        auto isTrivial = [](const MOPTriangle& t) -> bool {
            return (t.materialIdInside == t.materialIdOutside) &&
                   (t.surfaceId == 0);
        };

        struct PlaneKey {
            float cx, cy, cz;   // centroid
            float nx, ny, nz;   // unit normal
        };
        std::vector<PlaneKey> nonTrivialKeys;
        size_t nonTrivCount = 0, trivCount = 0;
        for (const auto& dv : deferredVolumes) {
            for (const auto& t : dv.triangles) {
                if (isTrivial(t)) { ++trivCount; continue; }
                ++nonTrivCount;
                PlaneKey k;
                k.cx = (t.v0x + t.v1x + t.v2x) / 3.0f;
                k.cy = (t.v0y + t.v1y + t.v2y) / 3.0f;
                k.cz = (t.v0z + t.v1z + t.v2z) / 3.0f;
                k.nx = t.nx; k.ny = t.ny; k.nz = t.nz;
                nonTrivialKeys.push_back(k);
            }
        }

        const float CENTROID_EPS_MM = 0.01f;            // 10 µm
        const float CENTROID_EPS2 = CENTROID_EPS_MM * CENTROID_EPS_MM;
        const float NORMAL_DOT_THRESHOLD = 0.999f;       // ~2.5°

        size_t totalDropped = 0;
        for (auto& dv : deferredVolumes) {
            std::vector<MOPTriangle> kept;
            kept.reserve(dv.triangles.size());
            for (const auto& t : dv.triangles) {
                if (!isTrivial(t)) {
                    kept.push_back(t);
                    continue;
                }
                float tcx = (t.v0x + t.v1x + t.v2x) / 3.0f;
                float tcy = (t.v0y + t.v1y + t.v2y) / 3.0f;
                float tcz = (t.v0z + t.v1z + t.v2z) / 3.0f;
                bool coincident = false;
                for (const auto& k : nonTrivialKeys) {
                    float dx = tcx - k.cx;
                    float dy = tcy - k.cy;
                    float dz = tcz - k.cz;
                    if (dx*dx + dy*dy + dz*dz > CENTROID_EPS2) continue;
                    float dotN = t.nx * k.nx + t.ny * k.ny + t.nz * k.nz;
                    if (std::fabs(dotN) < NORMAL_DOT_THRESHOLD) continue;
                    coincident = true;
                    break;
                }
                if (coincident) {
                    ++totalDropped;
                } else {
                    kept.push_back(t);
                }
            }
            dv.triangles = std::move(kept);
        }

        if (m_logLevel >= 1) {
            std::cout << "[TsGPU] Fix 81 v2 coincidence filter: scanned "
                      << nonTrivCount << " non-trivial + " << trivCount
                      << " trivial faces, dropped " << totalDropped
                      << " trivial-coincident triangle(s)" << std::endl;
        }

        // ============================================================
        // Fix 82 (2026-04-25): Sibling adjacency rewriting.
        //
        // 5명 합의 architecture (Opticks-aligned). G4Navigator
        // .LocateGlobalPointAndUpdateTouchableHandle() 동등 의미.
        //
        // 문제:
        //   기존 mesh 추출은 각 triangle 의 materialIdOutside 를 부모 볼륨
        //   매물로 하드코딩. 두 sibling 볼륨이 인접한 face 에서는 양쪽 모두
        //   materialOut=parent 가 되어, GPU boundary processor 가 실제 인접
        //   sibling 매물을 모르고 부모(보통 World Air) 로 Fresnel 계산. 결과:
        //   같은 n 인접 sibling (예: BC408 / BC408_Black) 도 BC408->Air
        //   Fresnel 잘못 적용 (50% 반사) → R 광자 일부가 ScRefl 로 못 들어감.
        //
        // 해결:
        //   모든 non-trivial triangle 쌍 검사 → coincident pair (centroid <
        //   10 µm + |dot(n_A, n_B)| > 0.999) 발견 시, 각각의 materialIdOutside
        //   를 상대 볼륨의 materialIdInside 로 재작성. Geant4 G4Navigator 가
        //   인접 볼륨을 동적 조회해서 얻는 결과와 동일.
        //
        // 한계:
        //   - Border surface (G4LogicalBorderSurface) 의 surfaceId 매칭은
        //     아직 미구현 (m_borderSurfacePVs 가 PV pointer key 라 별도 구현
        //     필요). 우선은 매물만 정확히 매칭.
        //   - Dedupe 없음 (양쪽 triangle 모두 keep). Metal RT 가 둘 중 어느
        //     쪽을 hit 해도 boundary attribute 일관됨. 비결정성 우려시 후속
        //     pass 에서 dedupe 추가.
        // ============================================================
        {
            int matRewritten = 0;
            int surfPropagated = 0;
            int dedupedCount = 0;
            // FaceRef: 모든 non-trivial triangle 정보 + 평면 signature.
            // Sci 와 Wrap 처럼 인접 volume 이 OPPOSITE diagonal 로 face 를 분할하면
            // triangle centroid 는 일치 안함 (다른 위치). 대신 PLANE 단위로 매칭.
            // Plane signature: canonical normal (first non-zero positive) + signed
            // distance from origin (= dot(any vertex, canonical normal)).
            struct FaceRef {
                size_t volIdx;
                size_t triIdx;
                uint32_t matInside;
                uint32_t surfaceId;
                float cx, cy, cz;
                float nx, ny, nz;
                // Plane signature
                float planeNx, planeNy, planeNz;  // canonical normal
                float planeD;                      // signed distance from origin
                // Centroid projected onto face plane (2D position within face)
                // — used for AABB overlap check between coincident-plane faces
                float ax, ay;                      // axis U projection
                // Triangle vertex bounds in plane (AABB)
                float minU, maxU, minV, maxV;
            };
            std::vector<FaceRef> allFaces;
            allFaces.reserve(nonTrivCount);
            for (size_t v = 0; v < deferredVolumes.size(); v++) {
                auto& dv = deferredVolumes[v];
                for (size_t t = 0; t < dv.triangles.size(); t++) {
                    const auto& T = dv.triangles[t];
                    if (T.materialIdInside == T.materialIdOutside &&
                        T.surfaceId == 0) continue;  // skip trivial
                    FaceRef f;
                    f.volIdx = v;
                    f.triIdx = t;
                    f.matInside = T.materialIdInside;
                    f.surfaceId = T.surfaceId;
                    f.cx = (T.v0x + T.v1x + T.v2x) / 3.0f;
                    f.cy = (T.v0y + T.v1y + T.v2y) / 3.0f;
                    f.cz = (T.v0z + T.v1z + T.v2z) / 3.0f;
                    f.nx = T.nx; f.ny = T.ny; f.nz = T.nz;
                    // Canonicalize normal: first non-zero component positive
                    float canonSign = 1.0f;
                    if (std::fabs(f.nx) > 0.5f) canonSign = (f.nx > 0) ? 1.0f : -1.0f;
                    else if (std::fabs(f.ny) > 0.5f) canonSign = (f.ny > 0) ? 1.0f : -1.0f;
                    else canonSign = (f.nz > 0) ? 1.0f : -1.0f;
                    f.planeNx = f.nx * canonSign;
                    f.planeNy = f.ny * canonSign;
                    f.planeNz = f.nz * canonSign;
                    // Signed distance of plane from origin (signed by canonical normal)
                    f.planeD = T.v0x*f.planeNx + T.v0y*f.planeNy + T.v0z*f.planeNz;
                    // 2D AABB on face plane (using axes orthogonal to canonical normal)
                    // For axis-aligned face, these are simply the other 2 coordinates.
                    auto coord2D = [&](float x, float y, float z, float& u, float& v) {
                        if (std::fabs(f.planeNx) > 0.5f) { u = y; v = z; }
                        else if (std::fabs(f.planeNy) > 0.5f) { u = x; v = z; }
                        else { u = x; v = y; }
                    };
                    float u0, v0, u1, v1, u2, v2;
                    coord2D(T.v0x, T.v0y, T.v0z, u0, v0);
                    coord2D(T.v1x, T.v1y, T.v1z, u1, v1);
                    coord2D(T.v2x, T.v2y, T.v2z, u2, v2);
                    f.minU = std::min({u0, u1, u2});
                    f.maxU = std::max({u0, u1, u2});
                    f.minV = std::min({v0, v1, v2});
                    f.maxV = std::max({v0, v1, v2});
                    f.ax = (f.minU + f.maxU) * 0.5f;
                    f.ay = (f.minV + f.maxV) * 0.5f;
                    allFaces.push_back(f);
                }
            }

            // 각 face A 에 대해 다른 볼륨의 coincident face B 검색.
            // dedup 정책 (Opticks 수렴):
            //   - Coincident pair 발견 시 surface 가 있는 쪽 (B.surfaceId != 0
            //     이고 A.surfaceId == 0) 을 keep, A 는 dedupe 로 mark.
            //   - 둘 다 surface 있거나 둘 다 없으면 lower volIdx 를 keep.
            //   - Keep 되는 triangle 은 matIn/matOut 를 정확한 boundary identity
            //     (matIn = own volume, matOut = adjacent sibling) 로 갖춤.
            // 이렇게 하면 Metal RT 에서 단일 triangle 만 hit → mirror surface
            // 등 boundary attribute 가 결정적으로 적용됨.
            std::vector<std::vector<bool>> dedupeMark(deferredVolumes.size());
            for (size_t v = 0; v < deferredVolumes.size(); v++) {
                dedupeMark[v].resize(deferredVolumes[v].triangles.size(), false);
            }

            // Plane-based matching (face-level, not triangle-level).
            //   Sci 와 Wrap 처럼 인접 sibling 이 OPPOSITE diagonal 로 face 분할
            //   하면 triangle centroid 가 일치 안함 → 잘못된 mismatch.
            //   대신: PLANE signature (canonical normal + signed offset) 일치 +
            //   2D AABB overlap 으로 face level coincidence 판정.
            const float PLANE_NORMAL_EPS = 1e-3f;
            const float PLANE_OFFSET_EPS_MM = 0.1f;
            const float AABB_OVERLAP_EPS_MM = 1.0f;
            for (size_t i = 0; i < allFaces.size(); i++) {
                const auto& A = allFaces[i];
                if (dedupeMark[A.volIdx][A.triIdx]) continue;  // already removed
                for (size_t j = 0; j < allFaces.size(); j++) {
                    if (i == j) continue;
                    const auto& B = allFaces[j];
                    if (A.volIdx == B.volIdx) continue;
                    if (dedupeMark[B.volIdx][B.triIdx]) continue;
                    // Same plane: canonical normal match + signed distance match
                    if (std::fabs(A.planeNx - B.planeNx) > PLANE_NORMAL_EPS) continue;
                    if (std::fabs(A.planeNy - B.planeNy) > PLANE_NORMAL_EPS) continue;
                    if (std::fabs(A.planeNz - B.planeNz) > PLANE_NORMAL_EPS) continue;
                    if (std::fabs(A.planeD - B.planeD) > PLANE_OFFSET_EPS_MM) continue;
                    // Same plane confirmed. 2D AABB overlap check (face area, not edge).
                    float overlapU = std::min(A.maxU, B.maxU) - std::max(A.minU, B.minU);
                    float overlapV = std::min(A.maxV, B.maxV) - std::max(A.minV, B.minV);
                    if (overlapU < AABB_OVERLAP_EPS_MM ||
                        overlapV < AABB_OVERLAP_EPS_MM) continue;

                    // Sibling border surface lookup (정석 fix, 2026-04-26):
                    // G4LogicalBorderSurface(pvA, pvB) 정의되어 있으면 sibling face
                    // 에 surface 적용. 양방향 모두 검사 — Geant4 G4OpBoundaryProcess
                    // 가 (preStep_pv, postStep_pv) lookup 시 directional 매치를
                    // 사용하지만 BVH triangle 의 surfaceId 는 단일 값이라 단순화.
                    G4VPhysicalVolume* pvA = deferredVolumes[A.volIdx].pv;
                    G4VPhysicalVolume* pvB = deferredVolumes[B.volIdx].pv;
                    uint32_t siblingBorderSurfId = lookupBorderSurfId(pvA, pvB);

                    // Coincident pair (A, B) found. Decide which to keep.
                    // Rule: surface-bearing wins; tie → lower volIdx wins.
                    // 정석 fix: surfaceId 결정 시 sibling border surface 도 후보.
                    uint32_t effectiveA_surf = A.surfaceId != 0 ? A.surfaceId : siblingBorderSurfId;
                    uint32_t effectiveB_surf = B.surfaceId != 0 ? B.surfaceId : siblingBorderSurfId;
                    bool keepA;
                    if (effectiveA_surf != 0 && effectiveB_surf == 0) keepA = true;
                    else if (effectiveA_surf == 0 && effectiveB_surf != 0) keepA = false;
                    else keepA = (A.volIdx < B.volIdx);

                    auto& T_A = deferredVolumes[A.volIdx].triangles[A.triIdx];
                    auto& T_B = deferredVolumes[B.volIdx].triangles[B.triIdx];

                    // 정석 fix (2026-04-26): rectangle 2 tri 중 두 번째 tri 누락 fix.
                    // Mesh extraction (CreatePolyhedron / G4TessellatedSolid) 이 G4Box face
                    // (rectangle) 를 2 triangle 로 분할 → Fix 82 loop 가 face 단위 가
                    // 아니라 triangle 단위로 처리하면 같은 face 의 두 번째 tri 가 partner
                    // (이미 dedupeMark 됨) 못 찾아 keep but matOut=Air (default) 잔존.
                    // 결과: 50% photon 이 BVH hit 시 boundary skip → wrap 안 통과 → fluence
                    // over (U3v1side: GPU 3.5% over, post-medium fluence 누적).
                    // Fix: matOut update + surfaceId propagate 시 keep 측 volume 의 same
                    // plane (canonical normal + signed offset) tri 모두 동시 update.
                    auto propagateToSamePlane = [&](size_t keepVolIdx, uint32_t newMatOut,
                                                    uint32_t newSurfId, float keepNx, float keepNy,
                                                    float keepNz, float keepD) {
                        for (auto& other : allFaces) {
                            if (other.volIdx != keepVolIdx) continue;
                            if (dedupeMark[other.volIdx][other.triIdx]) continue;
                            if (std::fabs(other.planeNx - keepNx) > PLANE_NORMAL_EPS) continue;
                            if (std::fabs(other.planeNy - keepNy) > PLANE_NORMAL_EPS) continue;
                            if (std::fabs(other.planeNz - keepNz) > PLANE_NORMAL_EPS) continue;
                            if (std::fabs(other.planeD - keepD) > PLANE_OFFSET_EPS_MM) continue;
                            auto& T_other = deferredVolumes[other.volIdx].triangles[other.triIdx];
                            if (T_other.materialIdOutside != newMatOut) {
                                T_other.materialIdOutside = newMatOut;
                                matRewritten++;
                            }
                            if (T_other.surfaceId == 0 && newSurfId != 0) {
                                T_other.surfaceId = newSurfId;
                                surfPropagated++;
                            }
                        }
                    };

                    if (keepA) {
                        // A keeps; rewrite matOut & surface propagate from B
                        if (T_A.materialIdOutside != B.matInside) {
                            T_A.materialIdOutside = B.matInside;
                            matRewritten++;
                        }
                        if (T_A.surfaceId == 0 && effectiveA_surf != 0) {
                            T_A.surfaceId = effectiveA_surf;
                            surfPropagated++;
                        }
                        // Propagate to same-plane sibling tri in same volume
                        propagateToSamePlane(A.volIdx, B.matInside,
                                             effectiveA_surf,
                                             A.planeNx, A.planeNy, A.planeNz, A.planeD);
                        dedupeMark[B.volIdx][B.triIdx] = true;
                    } else {
                        if (T_B.materialIdOutside != A.matInside) {
                            T_B.materialIdOutside = A.matInside;
                            matRewritten++;
                        }
                        if (T_B.surfaceId == 0 && effectiveB_surf != 0) {
                            T_B.surfaceId = effectiveB_surf;
                            surfPropagated++;
                        }
                        propagateToSamePlane(B.volIdx, A.matInside,
                                             effectiveB_surf,
                                             B.planeNx, B.planeNy, B.planeNz, B.planeD);
                        dedupeMark[A.volIdx][A.triIdx] = true;
                    }
                    dedupedCount++;
                    break;  // A processed, move on
                }
            }

            // Apply dedupe: drop marked triangles
            for (size_t v = 0; v < deferredVolumes.size(); v++) {
                auto& dv = deferredVolumes[v];
                std::vector<MOPTriangle> kept;
                kept.reserve(dv.triangles.size());
                for (size_t t = 0; t < dv.triangles.size(); t++) {
                    if (!dedupeMark[v][t]) kept.push_back(dv.triangles[t]);
                }
                dv.triangles = std::move(kept);
            }

            if (m_logLevel >= 1) {
                std::cout << "[TsGPU] Fix 82 sibling adjacency: "
                          << matRewritten << " matOut + "
                          << surfPropagated << " surfaceId propagated, "
                          << dedupedCount << " coincident pair deduped"
                          << std::endl;
            }
        }

        // 누적된 (필터링된) 볼륨 메시를 엔진에 일괄 push.
        for (const auto& dv : deferredVolumes) {
            if (dv.triangles.empty()) continue;
            MOPEngine_AddVolumeMesh(m_engine, dv.triangles.data(),
                                    (uint32_t)dv.triangles.size(), dv.volumeId);

            if (m_logLevel >= 3) {
                float minX=1e10, maxX=-1e10, minY=1e10, maxY=-1e10, minZ=1e10, maxZ=-1e10;
                for (const auto& t : dv.triangles) {
                    minX = std::min({minX, t.v0x, t.v1x, t.v2x});
                    maxX = std::max({maxX, t.v0x, t.v1x, t.v2x});
                    minY = std::min({minY, t.v0y, t.v1y, t.v2y});
                    maxY = std::max({maxY, t.v0y, t.v1y, t.v2y});
                    minZ = std::min({minZ, t.v0z, t.v1z, t.v2z});
                    maxZ = std::max({maxZ, t.v0z, t.v1z, t.v2z});
                }
                std::cout << "[TsGPU] Volume '" << dv.ownerName
                          << "' volId=" << dv.volumeId
                          << " matIn=" << dv.matIn << " matOut=" << dv.matOut
                          << " surfId=" << dv.surfId
                          << " tris=" << dv.triangles.size()
                          << " bbox=(" << minX << "," << minY << "," << minZ
                          << ")-(" << maxX << "," << maxY << "," << maxZ << ")"
                          << std::endl;
            }
        }
    }
}

// ============================================================
// Geant4 물질 → MOP material ID 매핑
// ============================================================
uint32_t TsGPUOpticalPhysics::GetMOPMaterialId(const G4Material* g4mat) const {
    if (!g4mat) return 0;
    std::string name = g4mat->GetName();
    auto it = m_materialMap.find(name);
    if (it != m_materialMap.end()) return it->second;
    // m8 수정: thread-safe 로그 제한
    if (m_logLevel >= 4) {
        std::cerr << "[TsGPU] Warning: Unmapped material '" << name << "'" << std::endl;
    }
    return 0;
}

// Fix 39: thread_local 정의
thread_local TsGPUOpticalPhysics::ThreadLocalGensteps TsGPUOpticalPhysics::t_localGensteps;
thread_local TsGPUOpticalPhysics::ThreadLocalEventState TsGPUOpticalPhysics::t_eventState;
// SF1 deficit fix: Run-lifetime accumulator (cross-thread, never reset within run)
std::vector<MOPSurfaceHit> TsGPUOpticalPhysics::s_accumulatedSurfaceHits;
// 2026-05-19: Run-lifetime TransitHit accumulator (PhaseSpace scorer 용)
std::vector<MOPHit> TsGPUOpticalPhysics::s_accumulatedTransitHits;
bool TsGPUOpticalPhysics::s_transitAccumulationEnabled = false;

// ============================================================
// 이벤트 처리 (Fix 39: MT 안전 — 스레드별 genstep 수집)
// ============================================================
void TsGPUOpticalPhysics::BeginOfEvent() {
    // 현재 스레드의 genstep만 초기화 (다른 스레드에 영향 없음)
    t_localGensteps.gensteps.clear();
    t_localGensteps.pendingPhotons = 0;
    t_localGensteps.sent = false;
    t_eventState.eventAlreadyProcessed = false;
    t_eventState.lastHitCount = 0;

    // Primary opticalphoton 카운터 리셋 — 새 batch 첫 이벤트인 경우만.
    // 2026-05-18 MT race fix: 이전 (m_batchEventCount==0) check 은 mutex 안
    // 일관 read 지만 m_batchEventCount 가 EndOfEvent 까지 0 인 채로 머무름 →
    // 여러 thread 의 BeginOfEvent 가 동시 reset 호출 → 이미 push 된 primary
    // photon storage overwrite (race 의 진짜 source).
    // Fix: m_isFirstEventOfBatch flag (FlushBatch 가 true set, BeginOfEvent
    // 가 atomic swap 으로 한 번만 reset).
    // ForceRepropagate 가 같은 primary set 을 재사용하기 위해 FlushBatch
    // 직후 reset 하지 않고 다음 batch 시작 시점에 reset 유지.
    if (m_engine) {
        std::lock_guard<std::recursive_mutex> lock(m_gpuMutex);
        if (m_isFirstEventOfBatch) {
            MOPEngine_ResetPrimaryPhotons(m_engine);
            m_isFirstEventOfBatch = false;
        }
    }
}

uint32_t TsGPUOpticalPhysics::EndOfEvent() {
    // GPU 전파는 직렬화 (한 번에 하나의 이벤트만 GPU 사용)
    std::lock_guard<std::recursive_mutex> lock(m_gpuMutex);
    if (!m_engine || !m_gpuEnabled) return 0;

    // 2026-05-12: BEAM mode — 첫 EndOfEvent 호출 시 BEAM genstep push.
    // 1B 같은 대형 N 은 GPU batch limit (MOP_MAX_PHOTONS_PER_BATCH=128M) 으로 split.
    // 100M 청크로 multiple FlushBatch 호출 → 메모리 안전 + 전체 N 처리.
    if (m_beamMode && (m_beamPhotonCount > 0 || !m_extraBeamGensteps.empty())) {
        const int64_t CHUNK = 5000000;  // 5M per GPU dispatch
        // 2026-05-17 multi-beam: primary beam (m_beamGenstep/m_beamPhotonCount)
        // + 추가 beam (m_extraBeamGensteps/m_extraBeamPhotonCounts) 모두 같은 chunking.
        struct BeamItem { MOPGenstep gs; int64_t n; };
        std::vector<BeamItem> beams;
        if (m_beamPhotonCount > 0) {
            beams.push_back({m_beamGenstep, m_beamPhotonCount});
        }
        for (size_t i = 0; i < m_extraBeamGensteps.size(); ++i) {
            if (m_extraBeamPhotonCounts[i] > 0) {
                beams.push_back({m_extraBeamGensteps[i], m_extraBeamPhotonCounts[i]});
            }
        }
        // 재진입 방지
        m_beamPhotonCount = 0;
        m_beamGenstep.numPhotons = 0;
        m_extraBeamGensteps.clear();
        m_extraBeamPhotonCounts.clear();

        int64_t totalPhotons = 0;
        for (auto& b : beams) totalPhotons += b.n;
        int64_t totalChunks = 0;
        for (auto& b : beams) totalChunks += (b.n + CHUNK - 1) / CHUNK;
        if (m_logLevel >= 1) {
            std::cout << "[TsGPU] BEAM mode: " << totalPhotons << " photons across "
                      << beams.size() << " beam(s) → " << totalChunks
                      << " GPU chunks × " << CHUNK << std::endl;
        }
        for (auto& b : beams) {
            MOPGenstep baseGS = b.gs;
            int64_t remaining = b.n;
            int64_t numChunks = (remaining + CHUNK - 1) / CHUNK;
            for (int64_t c = 0; c < numChunks; c++) {
                int64_t chunkN = std::min<int64_t>(remaining, CHUNK);
                MOPGenstep gs = baseGS;
                gs.numPhotons = (uint32_t)chunkN;
                m_batchGensteps.clear();
                m_batchGensteps.push_back(gs);
                m_batchPendingPhotons = (uint32_t)chunkN;
                m_batchEventCount = m_batchSize;  // force flush
                FlushBatch();
                remaining -= chunkN;
            }
        }
    }

    auto& local = t_localGensteps;

    // 이미 이 이벤트에서 GPU 전파가 완료된 경우:
    if (t_eventState.eventAlreadyProcessed) {
        if (local.gensteps.empty()) {
            return t_eventState.lastHitCount;
        }
        t_eventState.eventAlreadyProcessed = false;
    }

    if (local.gensteps.empty()) {
        t_eventState.lastHitCount = 0;
        t_eventState.eventAlreadyProcessed = true;
        // Fix 63: 빈 이벤트도 배치 카운트에 포함
        m_batchEventCount++;
        // Primary opticalphoton beam의 경우 gensteps가 0이지만 엔진에는
        // 이미 primary photon이 push된 상태일 수 있음 → batch가 차면 dispatch.
        bool hasPendingPrimaries =
            (m_engine && MOPEngine_GetPrimaryPhotonCount(m_engine) > 0);
        if (m_batchEventCount >= m_batchSize &&
            (!m_batchGensteps.empty() || hasPendingPrimaries)) {
            FlushBatch();
        }
        return 0;
    }

    // Fix 63: 이벤트 배칭 — genstep을 누적하고, 배치가 차면 한 번에 dispatch
    m_batchGensteps.insert(m_batchGensteps.end(),
                           local.gensteps.begin(), local.gensteps.end());
    m_batchPendingPhotons += local.pendingPhotons;
    m_batchEventCount++;

    local.gensteps.clear();
    local.pendingPhotons = 0;
    local.sent = true;

    // 배치가 차지 않았으면 아직 dispatch하지 않음
    if (m_batchEventCount < m_batchSize) {
        t_eventState.lastHitCount = 0;
        t_eventState.cachedHits.clear();
        t_eventState.cachedTransitHits.clear();
        t_eventState.eventAlreadyProcessed = true;
        return 0;
    }

    // === 배치 가득 참 → GPU dispatch ===
    FlushBatch();
    return t_eventState.lastHitCount;
}

void TsGPUOpticalPhysics::FlushBatch() {
    // Primary photon이 비어있고 gensteps도 비어있으면 dispatch 의미 없음.
    bool hasPrimaries =
        (m_engine && MOPEngine_GetPrimaryPhotonCount(m_engine) > 0);
    if (m_batchGensteps.empty() && !hasPrimaries) {
        m_batchEventCount = 0;
        return;
    }

    if (m_logLevel >= 2) {
        std::cout << "[TsGPU] FlushBatch: " << m_batchEventCount << " events, "
                  << m_batchGensteps.size() << " gensteps, "
                  << m_batchPendingPhotons << " photons" << std::endl;
    }

    // 2026-05-12 fix: photon buffer cap (MOP_MAX_PHOTONS_PER_BATCH = 128M) 안전
    // multi-chunk dispatch. 이전 single-dispatch 는 cap 초과 시 silent truncate
    // (engine Propagate() 가 totalPhotons > cap 일 때 genstepPhotons clamp →
    //  cap 초과 광자 silent drop). batch=10000 시 sim_light 시나리오에서 800M
    //  photon 중 134M 만 처리 → fluence sum 33% under. cap-aware chunk split 으로
    //  fix.
    //
    // Chunk size = 5M (실측 sweet spot): batch=5 의 정상 dispatch 가 400K photons
    // 에서 127ns/photon, batch=10000 single 134M dispatch 가 695ns/photon — 5.5×
    // 차이. 작은 chunk 는 GPU occupancy + cache locality 더 좋음. 5M 은 batch
    // size 와 무관하게 dispatch 당 GPU efficient 영역 보장.
    const uint32_t CHUNK_CAP = 5u * 1000u * 1000u;
    uint32_t hasPrimariesCount = hasPrimaries ?
        (m_engine ? MOPEngine_GetPrimaryPhotonCount(m_engine) : 0u) : 0u;

    if (m_batchPendingPhotons + hasPrimariesCount <= CHUNK_CAP) {
        // === 정상 path: single dispatch ===
        // Fix 71: 파이프라이닝 — 이전 GPU 작업 완료 대기 및 결과 수확
        HarvestPendingGPUResults();
        MOPEngine_ResetGensteps(m_engine);
        for (const auto& gs : m_batchGensteps) {
            MOPEngine_AddGenstep(m_engine, &gs);
        }
        // Primary opticalphoton은 ProcessStep/PostStepDoIt에서 이미 엔진에 push됨.
        // FlushBatch 종료 시 reset하지 않음 — 동일 batch에서 여러 scorer가
        // ForceRepropagate로 같은 primary photon set을 재사용할 수 있도록.
        MOPEngine_Propagate(m_engine);
        m_gpuPipelinePending = true;
    } else {
        // === Multi-chunk: genstep group 으로 split ===
        // 단일 genstep 의 numPhotons 가 CHUNK_CAP 초과시 잘림 (현재 사용 시나리오
        // 에서는 발생 안 함 — Beam genstep 은 EndOfEvent 에서 별도 chunk loop;
        // Scintillation/Cerenkov per-step 광자 수 ≪ 128M).
        if (m_logLevel >= 2) {  // 2026-05-17: per-chunk verbose → logLevel >= 2 (1B = 200 chunks 면 너무 많음)
            std::cout << "[TsGPU] FlushBatch multi-chunk: "
                      << m_batchPendingPhotons << " photons (cap "
                      << CHUNK_CAP << ") — split required" << std::endl;
        }
        std::vector<MOPGenstep> chunkGS;
        uint32_t chunkPhotons = 0;
        uint32_t chunksDispatched = 0;
        auto dispatchChunk = [&]() {
            if (chunkGS.empty()) return;
            HarvestPendingGPUResults();
            MOPEngine_ResetGensteps(m_engine);
            for (const auto& g : chunkGS) MOPEngine_AddGenstep(m_engine, &g);
            MOPEngine_Propagate(m_engine);
            m_gpuPipelinePending = true;
            chunkGS.clear();
            chunkPhotons = 0;
            chunksDispatched++;
        };
        for (const auto& gs : m_batchGensteps) {
            if (chunkPhotons + gs.numPhotons > CHUNK_CAP && !chunkGS.empty()) {
                dispatchChunk();
            }
            chunkGS.push_back(gs);
            chunkPhotons += gs.numPhotons;
        }
        dispatchChunk();  // 마지막 chunk
        if (m_logLevel >= 2) {  // 2026-05-17: per-chunk dispatch summary → logLevel >= 2
            std::cout << "[TsGPU] FlushBatch multi-chunk: "
                      << chunksDispatched << " chunks dispatched" << std::endl;
        }
    }

    // 배치 리셋 (CPU는 즉시 다음 이벤트 처리 가능)
    m_batchGensteps.clear();
    m_batchPendingPhotons = 0;
    m_batchEventCount = 0;
    m_isFirstEventOfBatch = true;  // 2026-05-18 MT race fix: 다음 batch BeginOfEvent 가 ResetPrimary
    t_eventState.eventAlreadyProcessed = true;
}

// Fix C1: Run 종료 시점의 잔여 배치 flush + 결과 수확 (스코어러 훅용)
// EventBatchSize와 NumberOfHistoriesInRun이 정확히 배수가 아닐 때
// 남은 (N%BatchSize) 이벤트의 genstep이 버퍼에 갇혀 소실되는 문제 해결.
//
// 정석 fix (2026-04-26): hasPrimaries 만으로 FlushBatch trigger 시 직전 batch
// 의 stale primary photon (FlushBatch 후 ResetPrimaryPhotons 가 deferred 되어
// 다음 BeginOfEvent 까지 남아있음) 도 trigger 됨 → 같은 photon set 이 두 번
// dispatch 되어 +1000 photon over-generation (1.0% sum bias).
//
// 진짜 잔여 batch 는 m_batchEventCount > 0 (아직 dispatch 안 된 events)
// 또는 m_batchGensteps 가 차있는 경우. hasPrimaries 만으로는 trigger 안 함.
uint32_t TsGPUOpticalPhysics::FinalizePendingBatch() {
    std::lock_guard<std::recursive_mutex> lock(m_gpuMutex);
    if (!m_engine || !m_gpuEnabled) return 0;

    // 아직 dispatch 안 된 잔여 배치만 처리 (stale primary photon 제외).
    if (!m_batchGensteps.empty() || m_batchEventCount > 0) {
        if (m_logLevel >= 1) {
            std::cout << "[TsGPU] FinalizePendingBatch: flushing "
                      << m_batchEventCount << " events, "
                      << m_batchGensteps.size() << " gensteps"
                      << std::endl;
        }
        FlushBatch();  // 내부에서 HarvestPendingGPUResults 호출 후 새 dispatch
    }
    // 마지막 dispatch의 비동기 결과 회수 (스코어러가 transit hit 읽을 수 있도록)
    HarvestPendingGPUResults();
    return t_eventState.lastHitCount;
}

// Fix 71: GPU 결과 수확 (이전 FlushBatch에서 제출한 비동기 작업)
void TsGPUOpticalPhysics::HarvestPendingGPUResults() {
    if (!m_gpuPipelinePending) return;

    // GetHits/GetTransitHits 호출 시 내부적으로 WaitForGPU() 실행
    uint32_t transitCount = MOPEngine_GetLastTransitCount(m_engine);
    if (transitCount > 0) {
        t_eventState.cachedTransitHits.resize(transitCount);
        uint32_t cached = MOPEngine_GetTransitHits(m_engine,
            t_eventState.cachedTransitHits.data(), transitCount);
        t_eventState.cachedTransitHits.resize(cached);
        // 2026-05-19: PhaseSpace scorer 용 run-lifetime accumulator append
        // (cachedTransitHits 는 batch boundary 마다 cleared → run-wide dump 불가능)
        // 2026-05-23: PhaseSpace scorer 가 등록된 경우만 누적 (DDA Fluence-only 1B run 은
        // 무한 누적 → OOM. PhaseSpace 없으면 skip).
        if (s_transitAccumulationEnabled) {
            s_accumulatedTransitHits.insert(s_accumulatedTransitHits.end(),
                t_eventState.cachedTransitHits.begin(),
                t_eventState.cachedTransitHits.end());
        }
    } else {
        t_eventState.cachedTransitHits.clear();
    }

    // SF1 (2026-04-24): surface hit 캐시 수확 + accumulator append
    uint32_t surfaceCount = MOPEngine_GetLastSurfaceHitCount(m_engine);
    if (surfaceCount > 0) {
        t_eventState.cachedSurfaceHits.resize(surfaceCount);
        uint32_t cached = MOPEngine_GetSurfaceHits(m_engine,
            t_eventState.cachedSurfaceHits.data(), surfaceCount);
        t_eventState.cachedSurfaceHits.resize(cached);

        // SF1 deficit fix: Run-lifetime accumulator append
        // (cross-thread / multi-EndOfRun race 회피)
        s_accumulatedSurfaceHits.insert(s_accumulatedSurfaceHits.end(),
            t_eventState.cachedSurfaceHits.begin(),
            t_eventState.cachedSurfaceHits.end());
    } else {
        t_eventState.cachedSurfaceHits.clear();
    }

    // Bug A fix: cache 갱신 표시
    t_eventState.cacheVersion++;

    uint32_t hits = m_lastHitCount;
    if (hits > 0) {
        const uint32_t maxHits = 1000000;
        t_eventState.cachedHits.resize(std::min(hits, maxHits));
        uint32_t cached = MOPEngine_GetHits(m_engine,
            t_eventState.cachedHits.data(), (uint32_t)t_eventState.cachedHits.size());
        t_eventState.cachedHits.resize(cached);
    } else {
        t_eventState.cachedHits.clear();
    }
    t_eventState.lastHitCount = hits;

    if (m_logLevel >= 2) {
        std::cout << "[TsGPU] Pipeline harvest: " << hits << " hits, "
                  << transitCount << " transit" << std::endl;
    }

    m_gpuPipelinePending = false;

    // 2026-05-12 fix: harvest 직후 scorer callback 호출 — DDA dispatch 를
    // queue 에 enqueue. 이후 FlushBatch 가 propagation cmdBuf 를 enqueue 하면
    // queue FIFO 순서로 DDA 가 먼저 실행 → batch N 의 m_transitHitBuffer
    // 데이터 정확히 read. 이전 race (DDA 가 batch N+1 의 data 를 처리) 해소.
    // recursive_mutex 사용 — callback 안에서 RunGPUDDAAccumulateLocked 재진입 OK.
    for (auto& cb : m_harvestCallbacks) {
        if (cb) cb();
    }
}

void TsGPUOpticalPhysics::RegisterHarvestCallback(HarvestCallback cb) {
    std::lock_guard<std::recursive_mutex> lock(m_gpuMutex);
    m_harvestCallbacks.push_back(std::move(cb));
}

// 2026-05-17: 추가 beam genstep — m_extraBeamGensteps 에 누적, chunking 시 별도 dispatch.
void TsGPUOpticalPhysics::AddBeamGenstep(double cx, double cy, double cz,
                                          double dx, double dy, double dz,
                                          double energy_eV,
                                          double posCutoffX, double posCutoffY,
                                          double sigmaX, double sigmaY,
                                          double cutoffX, double cutoffY,
                                          int64_t numPhotons) {
    MOPGenstep gs = {};
    gs.posX = (float)cx; gs.posY = (float)cy; gs.posZ = (float)cz;
    gs.dirX = (float)dx; gs.dirY = (float)dy; gs.dirZ = (float)dz;
    gs.charge = (float)energy_eV;
    gs.genType = 3;
    gs.materialId = 0;
    gs.numPhotons = (numPhotons > 0) ? 1 : 0;
    gs.parentTrackId = 0;
    gs.betaInverse = (float)posCutoffX;
    gs.pmin = (float)posCutoffY;
    gs.pmax = (float)sigmaX;
    gs.maxCos = (float)sigmaY;
    gs.yieldRatio = (float)cutoffX;
    gs.edep = (float)cutoffY;
    gs.time = 0;
    gs.stepLength = 0;
    m_extraBeamGensteps.push_back(gs);
    m_extraBeamPhotonCounts.push_back(numPhotons);
    m_beamMode = true;
}

// 2026-05-12: GPU beam source — 1 genstep → numPhotons photons GPU dispatch.
// 파라미터 (Module 측에서 호출) 를 m_beamGenstep 에 저장. 첫 EndOfEvent 에 batch 에 push.
void TsGPUOpticalPhysics::SetBeamGenstep(double cx, double cy, double cz,
                                          double dx, double dy, double dz,
                                          double energy_eV,
                                          double posCutoffX, double posCutoffY,
                                          double sigmaX, double sigmaY,
                                          double cutoffX, double cutoffY,
                                          int64_t numPhotons,
                                          int     posShape,
                                          double  posSpreadX,
                                          double  posSpreadY,
                                          int     polMode,
                                          double  polX,
                                          double  polY,
                                          double  polZ,
                                          int     posDist,
                                          int     angDist,
                                          double  energySpreadEv,
                                          double  timeSpread,
                                          double  timeCutoff) {
    m_beamMode = true;
    m_beamPhotonCount = numPhotons;
    MOPGenstep& gs = m_beamGenstep;
    gs.posX = (float)cx; gs.posY = (float)cy; gs.posZ = (float)cz;
    gs.dirX = (float)dx; gs.dirY = (float)dy; gs.dirZ = (float)dz;
    gs.charge = (float)energy_eV;  // BEAM: charge = energy (eV)
    gs.genType = 3;  // MOP_GEN_BEAM
    gs.materialId = 0;
    // gs.numPhotons 은 chunking 루프에서 per-chunk size 로 세팅 (uint32 fit).
    gs.numPhotons = (numPhotons > 0) ? 1 : 0;  // sentinel non-zero (chunking 트리거)
    gs.parentTrackId = 0;
    gs.betaInverse = (float)posCutoffX;  // BEAM: pos cutoff X half-width (mm)
    gs.pmin = (float)posCutoffY;          // BEAM: pos cutoff Y half-width (mm)
    gs.pmax = (float)sigmaX;              // BEAM: ang sigma X (rad)
    gs.maxCos = (float)sigmaY;            // BEAM: ang sigma Y (rad)
    gs.yieldRatio = (float)cutoffX;       // BEAM: ang cutoff X (rad)
    gs.edep = (float)cutoffY;             // BEAM: ang cutoff Y (rad)
    gs.time = 0;
    gs.stepLength = 0;

    // 2026-05-17: BeamPosition/Polarization 옵션은 SimConfig 에 반영.
    // SetBeamGenstep 은 Module Init 시 (BuildGeometry 이전) 호출되므로 m_engine 이
    // 살아있음. SimConfig 는 BuildGeometry 후 SetConfig 로 GPU에 한번 더 갱신되지만,
    // 여기 미리 set 해두면 BuildGeometry 후 initial SetConfig 가 그대로 가져감.
    if (m_engine) {
        MOPSimConfig cfg = GetConfig();
        cfg.beamPosShape    = (uint32_t)((posShape == 1) ? 1u : 0u);
        cfg.beamPosSigmaX   = (float)posSpreadX;
        cfg.beamPosSigmaY   = (float)posSpreadY;
        // Polarization vector 정규화 (zero vector → random transverse 강제)
        double pmag = std::sqrt(polX*polX + polY*polY + polZ*polZ);
        if (polMode == 1 && pmag > 1e-10) {
            cfg.beamPolMode = 1;
            cfg.beamPolX = (float)(polX / pmag);
            cfg.beamPolY = (float)(polY / pmag);
            cfg.beamPolZ = (float)(polZ / pmag);
        } else {
            cfg.beamPolMode = 0;
            cfg.beamPolX = 0.f; cfg.beamPolY = 0.f; cfg.beamPolZ = 0.f;
        }
        // 2026-05-17 추가 필드
        cfg.beamPosDist        = (uint32_t)std::max(0, std::min(2, posDist));
        cfg.beamAngDist        = (uint32_t)std::max(0, std::min(3, angDist));  // 3=Isotropic(4π)
        cfg.beamEnergySpreadEv = (float)energySpreadEv;
        cfg.beamTimeSpread     = (float)timeSpread;
        cfg.beamTimeCutoff     = (float)timeCutoff;
        SetConfig(cfg);
    }
}

// ============================================================
// Fix 42c: 강제 재전파 (AABB 설정 후 이미 전파된 이벤트를 다시 전파)
// 엔진에는 EndOfEvent에서 전송한 genstep이 아직 남아 있음 (ResetGensteps 안 함)
// ============================================================
uint32_t TsGPUOpticalPhysics::ForceRepropagate() {
    if (!m_engine || !m_gpuEnabled) return 0;

    if (!t_eventState.eventAlreadyProcessed) {
        // 아직 전파 안 됨 — 정상적으로 EndOfEvent 호출
        return EndOfEvent();
    }

    std::lock_guard<std::recursive_mutex> lock(m_gpuMutex);

    if (m_logLevel >= 1) {
        std::cout << "[TsGPU] ForceRepropagate: re-propagating with updated config" << std::endl;
    }

    // 2026-05-03 임시 revert: ForceRepropagate harvest 가 GPU result 절반으로 줄임
    // (U5 z scorer regression). race 패치 잠시 disable, root cause 분리.

    // Fix 42c: 엔진에 이미 genstep이 있음 (EndOfEvent에서 전송됨, ResetGensteps 안 함)
    // 단, local.gensteps는 EndOfEvent에서 클리어되었으므로 재전송 불가
    // 엔진의 genstep 버퍼는 그대로 유지되므로 바로 Propagate

    // GPU 재전파
    uint32_t hits = MOPEngine_Propagate(m_engine);
    t_eventState.lastHitCount = hits;

    // 히트 캐시 갱신
    if (hits > 0) {
        const uint32_t maxHits = 1000000;
        t_eventState.cachedHits.resize(std::min(hits, maxHits));
        uint32_t cached = MOPEngine_GetHits(m_engine,
            t_eventState.cachedHits.data(), (uint32_t)t_eventState.cachedHits.size());
        t_eventState.cachedHits.resize(cached);
    } else {
        t_eventState.cachedHits.clear();
    }

    // Transit hit 캐시 갱신
    uint32_t transitCount = MOPEngine_GetLastTransitCount(m_engine);
    if (transitCount > 0) {
        t_eventState.cachedTransitHits.resize(transitCount);
        uint32_t cached = MOPEngine_GetTransitHits(m_engine,
            t_eventState.cachedTransitHits.data(), transitCount);
        t_eventState.cachedTransitHits.resize(cached);
        if (m_logLevel >= 1) {
            std::cout << "[TsGPU] ForceRepropagate: " << cached
                      << " transit hits with AABB" << std::endl;
        }
    } else {
        t_eventState.cachedTransitHits.clear();
    }

    // Bug A fix: cache 갱신 표시 (scorer 가 신선 데이터 인지)
    t_eventState.cacheVersion++;

    t_eventState.eventAlreadyProcessed = true;
    return hits;
}

// ============================================================
// SteppingAction: Genstep 수집
// ============================================================
void TsGPUOpticalPhysics::ProcessStep(const G4Step* step) {
    // Lock-free: AddGenstep이 원자적 인덱스를 사용하므로 뮤텍스 불필요
    if (!m_engine || !m_gpuEnabled || !m_initialized || !m_geometryBuilt) {
        // static thread_local int debugCount = 0;
        // if (debugCount++ < 3) {
        //     std::cout << "[TsGPU] ProcessStep early return: engine=" << (m_engine!=nullptr)
        //               << " gpu=" << m_gpuEnabled << " init=" << m_initialized
        //               << " geom=" << m_geometryBuilt << std::endl;
        // }
        return;
    }

    // 디버그: ProcessStep 호출 횟수 추적
    static thread_local uint32_t totalCalls = 0;
    static thread_local uint32_t scintCalls = 0;
    static thread_local uint32_t noMptCalls = 0;
    static thread_local uint32_t noEdepCalls = 0;
    static thread_local uint32_t noYieldCalls = 0;
    static thread_local double totalEdep = 0;
    // Fix 51 진단: G4 vs GPU 광자 수 비교
    static thread_local uint64_t g4ScintPhotonTotal = 0;
    static thread_local uint64_t gpuGenstepPhotonTotal = 0;
    totalCalls++;

    const G4Track* track = step->GetTrack();
    const G4ParticleDefinition* particle = track->GetParticleDefinition();

    // 광학 광자: scintillation/Cerenkov로 생성된 secondary는 GPU에서 처리하므로
    // 여기서는 무시.
    // 2026-05-13: primary opticalphoton (Beam source) push 도 GPUOpticalKillProcess
    // (TsGPUOpticalPhysicsModule.cc:97) 가 전담. 여기서 중복 push 하지 않음.
    // 이전 코드는 CollectPrimaryPhoton 가 두 곳 (KillProcess + 여기) 호출 → 단일
    // 광자 2× transit hit → fluence 2× over.
    if (particle == G4OpticalPhoton::OpticalPhotonDefinition()) {
        return;
    }

    // Fix 51 진단: G4Scintillation이 생성한 광학 광자 2차 입자 수 카운트
    {
        const std::vector<const G4Track*>* secondaries = step->GetSecondaryInCurrentStep();
        if (secondaries) {
            for (size_t si = 0; si < secondaries->size(); si++) {
                if ((*secondaries)[si]->GetParticleDefinition() == G4OpticalPhoton::OpticalPhotonDefinition()) {
                    g4ScintPhotonTotal++;
                }
            }
        }
    }

    // 하전 입자의 스텝에서 신틸레이션/체렌코프 genstep 수집
    G4StepPoint* preStep = step->GetPreStepPoint();

    const G4Material* material = preStep->GetMaterial();
    if (!material) return;

    G4MaterialPropertiesTable* mpt = material->GetMaterialPropertiesTable();
    if (!mpt) { noMptCalls++; return; }

    // 에너지 침적이 있는 경우 신틸레이션 체크
    G4double edep = step->GetTotalEnergyDeposit();
    if (edep > 0) {
        // 신틸레이션 Yield가 있는 재질인지 확인
        if (mpt->ConstPropertyExists("SCINTILLATIONYIELD")) {
            scintCalls++;
            totalEdep += edep;
        } else {
            noYieldCalls++;
        }
        uint64_t beforePending = t_localGensteps.pendingPhotons;
        CollectScintillationGenstep(step);
        gpuGenstepPhotonTotal += (t_localGensteps.pendingPhotons - beforePending);
    } else {
        noEdepCalls++;
    }

    // 주기적 디버그 출력 (2026-05-15: >= 1 → >= 2, default LogLevel 에서 per-step 스팸 방지)
    if (totalCalls % 5000 == 0 && m_logLevel >= 2) {
        std::cout << "[TsGPU-Debug] ProcessStep stats: total=" << totalCalls
                  << " scint=" << scintCalls << " noMPT=" << noMptCalls
                  << " noEdep=" << noEdepCalls << " noYield=" << noYieldCalls
                  << " totalEdep=" << totalEdep/CLHEP::MeV << " MeV"
                  << " | Fix51: G4photons=" << g4ScintPhotonTotal
                  << " GPUphotons=" << gpuGenstepPhotonTotal
                  << " ratio=" << (gpuGenstepPhotonTotal > 0 ?
                     (double)g4ScintPhotonTotal / gpuGenstepPhotonTotal : 0)
                  << std::endl;
    }

    // 하전 입자의 경우 체렌코프 체크
    if (track->GetDefinition()->GetPDGCharge() != 0) {
        CollectCerenkovGenstep(step);
    }

    // 중간 전파 제거: 모든 genstep은 EndOfEvent에서 일괄 전파
    // (중간 전파는 뮤텍스 경합 + GPU 동기화 오버헤드의 주요 원인이었음)
}

// ============================================================
// 신틸레이션 Genstep 수집
// ============================================================
void TsGPUOpticalPhysics::CollectScintillationGenstep(const G4Step* step) {
    const G4Track* track = step->GetTrack();
    G4StepPoint* preStep = step->GetPreStepPoint();
    const G4Material* material = preStep->GetMaterial();
    G4MaterialPropertiesTable* mpt = material->GetMaterialPropertiesTable();

    // 신틸레이션 Yield 확인
    if (!mpt->ConstPropertyExists("SCINTILLATIONYIELD")) return;
    G4double scintYield = mpt->GetConstProperty("SCINTILLATIONYIELD");
    if (scintYield <= 0) return;

    G4double edep = step->GetTotalEnergyDeposit();

    // Fix 51: G4EmSaturation과 동일한 Birks 보정 사용
    // 기존 방식: edep / (1 + kB * edep/stepLength) — 스텝별 요동 dE/dx 사용
    // G4 방식:   G4EmSaturation::VisibleEnergyDepositionAtAStep — NIST 정지능 테이블 사용
    // Jensen 부등식으로 인해 기존 방식이 체계적으로 더 높은 visible energy 생성
    G4double visibleEnergy = edep;
    {
        G4EmSaturation* emSaturation = G4LossTableManager::Instance()->EmSaturation();
        if (emSaturation) {
            visibleEnergy = emSaturation->VisibleEnergyDepositionAtAStep(step);
        } else {
            G4double stepLength = step->GetStepLength();
            G4double birksConstant = material->GetIonisation()->GetBirksConstant();
            if (birksConstant > 0 && stepLength > 0 && edep > 0) {
                G4double dEdx = edep / stepLength;
                visibleEnergy = edep / (1.0 + birksConstant * dEdx);
            }
        }
    }

    G4double numPhotons = scintYield * visibleEnergy;

    // M2 수정: 포아송/가우시안 통계 (Geant4 호환)
    G4double resScale = 1.0;
    if (mpt->ConstPropertyExists("RESOLUTIONSCALE")) {
        resScale = mpt->GetConstProperty("RESOLUTIONSCALE");
    }

    if (numPhotons > 10.0) {
        // 가우시안 근사
        G4double sigma = resScale * std::sqrt(numPhotons);
        numPhotons = std::max(0.0, G4RandGauss::shoot(numPhotons, sigma));
    } else if (numPhotons > 0.0) {
        // 포아송 분포
        numPhotons = (G4double)G4Poisson(numPhotons);
    }

    if (numPhotons < 1) return;

    // yieldRatio 추출
    G4double yieldRatio = 1.0;
    if (mpt->ConstPropertyExists("SCINTILLATIONYIELD1"))
        yieldRatio = mpt->GetConstProperty("SCINTILLATIONYIELD1");
    else if (mpt->ConstPropertyExists("YIELDRATIO"))
        yieldRatio = mpt->GetConstProperty("YIELDRATIO");

    // Fix 53: step length 추출 (위치 랜덤화 및 genstep 기록용)
    G4double stepLength = step->GetStepLength();

    // Genstep 생성
    MOPGenstep gs;
    memset(&gs, 0, sizeof(gs));

    G4ThreeVector pos = preStep->GetPosition();
    gs.posX = pos.x() / CLHEP::mm;  // Geant4 → mm
    gs.posY = pos.y() / CLHEP::mm;
    gs.posZ = pos.z() / CLHEP::mm;
    gs.time = preStep->GetGlobalTime() / CLHEP::ns;  // → ns

    G4ThreeVector dir = track->GetMomentumDirection();
    gs.dirX = dir.x();
    gs.dirY = dir.y();
    gs.dirZ = dir.z();

    gs.charge = track->GetDefinition()->GetPDGCharge();
    gs.genType = MOP_GEN_SCINTILLATION;
    gs.materialId = GetMOPMaterialId(material);
    // Fix 53: Geant4 호환 rounding — G4Scintillation은 G4int(sample + 0.5) 사용
    gs.numPhotons = (uint32_t)(numPhotons + 0.5);
    gs.parentTrackId = track->GetTrackID();
    gs.edep = edep / CLHEP::MeV;
    gs.stepLength = stepLength / CLHEP::mm;
    gs.yieldRatio = (float)yieldRatio;  // m1 수정: genstep에 yieldRatio 포함

    // Fix 39: 스레드 로컬 벡터에 추가 (lock-free)
    t_localGensteps.gensteps.push_back(gs);
    t_localGensteps.pendingPhotons += gs.numPhotons;

    if (m_logLevel >= 3) {
        static thread_local int gsDebug = 0;
        if (gsDebug++ < 10) {
            std::cout << "[TsGPU] Scintillation genstep: matId=" << gs.materialId
                      << " nPhotons=" << gs.numPhotons << " edep=" << gs.edep
                      << " yield=" << scintYield << " pending=" << t_localGensteps.pendingPhotons << std::endl;
        }
    }
}

// ============================================================
// 체렌코프 Genstep 수집
// ============================================================
void TsGPUOpticalPhysics::CollectCerenkovGenstep(const G4Step* step) {
    const G4Track* track = step->GetTrack();
    G4StepPoint* preStep = step->GetPreStepPoint();
    const G4Material* material = preStep->GetMaterial();
    G4MaterialPropertiesTable* mpt = material->GetMaterialPropertiesTable();

    // 굴절률 확인
    G4MaterialPropertyVector* rindex = mpt->GetProperty("RINDEX");
    if (!rindex) return;

    // β 계산 — G4Cerenkov.cc:170 정석: midpoint average (pre + post)/2.
    // 이전 preStep 단독 사용은 fast-stopping δ-ray step (preStep β >> postStep β)
    // 에서 systematic over count → hadron Cherenkov 5-10% G/C over 의 source.
    G4StepPoint* postStep = step->GetPostStepPoint();
    G4double betaPre  = preStep->GetBeta();
    G4double betaPost = postStep->GetBeta();
    G4double beta = 0.5 * (betaPre + betaPost);
    if (beta <= 0) return;
    G4double betaInverse = 1.0 / beta;


    // 체렌코프 조건: βn > 1
    G4double maxRI = rindex->GetMaxValue();
    if (betaInverse >= maxRI) return;

    // C2 수정: 광자 에너지 범위 — Geant4 내부 단위(MeV) → eV
    G4double pmin = rindex->GetMinEnergy();  // Geant4 내부 단위
    G4double pmax = rindex->GetMaxEnergy();

    // βn > 1인 범위로 에너지 범위 축소
    // 최소 에너지에서 n(Emin)*β > 1인지 확인
    G4double riAtPmin = rindex->Value(pmin);
    if (riAtPmin * beta < 1.0) {
        // 이진 탐색으로 임계 에너지 찾기
        G4double eLow = pmin, eHigh = pmax;
        for (int iter = 0; iter < 20; iter++) {
            G4double eMid = 0.5 * (eLow + eHigh);
            if (rindex->Value(eMid) * beta > 1.0)
                eHigh = eMid;
            else
                eLow = eMid;
        }
        pmin = eHigh;
    }

    // C3 수정: Frank-Tamm 공식으로 체렌코프 광자 수 계산 (Geant4 내부 단위 사용)
    G4double charge = track->GetDefinition()->GetPDGCharge();
    G4double stepLength = step->GetStepLength();

    // 수치 적분: ∫ sin²θ(E) dE over [pmin, pmax]
    // sin²θ = 1 - 1/(βn(E))²
    // 2026-06-12 정석 fix: G4Cerenkov::GetAverageNumberOfPhotons (G4Cerenkov.cc:504-573) 와
    //   bit-호환되도록 yield 적분을 CAI(Cerenkov Angle Integral) 방식 + midpoint beta 로 교체.
    //   이전: 20-step 균일 사다리꼴 ∫(1-βInv²/n²)dE + pre/post beta 평균. 임계 근처에서 G4 의
    //   CAI 부분bin 선형보간과 어긋나 sim_light 총 fluence +0.4% (GPU>CPU) 의 source (2026-06-12 진단).
    //   G4: CAI = ∫ 1/n² dE over RINDEX knots (BuildThePhysicsTable:173-202),
    //       integral = (Pmax-Pmin) - (CAImax - CAI(Pmin))·βInv²,  βInv = midpoint (위 betaInverse).
    //   (이전 "pre/post 평균이 정석" 주석은 오해 — G4 PostStepDoIt:251,265 는 midpoint β 의
    //    GetAverageNumberOfPhotons × stepLength 로 광자수 계산. pre/post 는 do-while 에너지샘플용.)
    G4double CAImax = 0.0, CAImin = 0.0;
    {
        std::size_t nKnots = rindex->GetVectorLength();
        G4double prevE  = rindex->Energy(0);
        G4double prevRI = (*rindex)[0];
        G4double cum    = 0.0;
        bool pminDone   = (pmin <= prevE);   // 임계가 첫 knot 이하 → 전 범위 방사, CAImin=0
        for (std::size_t k = 1; k < nKnots; ++k) {
            G4double Ek  = rindex->Energy(k);
            G4double RIk = (*rindex)[k];
            G4double newCum = cum + (Ek - prevE) * 0.5 *
                              (1.0 / (prevRI * prevRI) + 1.0 / (RIk * RIk));
            if (!pminDone && pmin <= Ek) {   // G4 CAI->Value(Pmin): 누적적분 선형보간
                CAImin = cum + (newCum - cum) * (pmin - prevE) / (Ek - prevE);
                pminDone = true;
            }
            cum = newCum;
            prevE = Ek; prevRI = RIk;
        }
        CAImax = cum;
        if (!pminDone) CAImin = CAImax;      // 임계가 마지막 knot 이상 (방사 없음)
    }
    G4double ge = CAImax - CAImin;
    G4double integral = (pmax - pmin) - ge * betaInverse * betaInverse;
    if (integral < 0.0) integral = 0.0;

    // N = α z² L ∫sin²θ dE / ℏc (모두 Geant4 내부 단위)
    // G4Cerenkov 정석: mean N̄ → Poisson sample.
    // 이전 deterministic round + (numPhotons<1) cutoff 는 작은 step 의 yield 를
    // systematic 으로 skip 함 (Bragg 근처/β 임계). Scintillation 과 동일하게 Poisson.
    G4double alpha = CLHEP::fine_structure_const;
    G4double meanNumPhotons = alpha * charge * charge * stepLength * integral / CLHEP::hbarc;

    if (meanNumPhotons <= 0.0) return;
    G4double numPhotons = (G4double)G4Poisson(meanNumPhotons);
    if (numPhotons < 1) return;
    numPhotons = std::min(numPhotons, (G4double)m_cerenkovMaxPhotons);

    MOPGenstep gs;
    memset(&gs, 0, sizeof(gs));

    G4ThreeVector pos = preStep->GetPosition();
    gs.posX = pos.x() / CLHEP::mm;
    gs.posY = pos.y() / CLHEP::mm;
    gs.posZ = pos.z() / CLHEP::mm;
    gs.time = preStep->GetGlobalTime() / CLHEP::ns;

    // 2026-05-18 정석 fix: G4Cerenkov.cc:155 와 동일하게 chord direction 사용.
    //   G4: p0 = aStep.GetDeltaPosition().unit()  ← chord direction (forward biased)
    //   이전 우리: track->GetMomentumDirection()  ← current momentum (multi-scatter
    //              후 sideways 가능). C12 200 MeV/u 의 δ-ray cherenkov 광자
    //              mean path 4.2% 짧음의 root cause (cone axis sideways biased).
    // deltaPos.mag() < ε (단일 point step) 시 fallback to momentum direction.
    G4ThreeVector dPos = postStep->GetPosition() - preStep->GetPosition();
    G4double dPosMag = dPos.mag();
    G4ThreeVector dir;
    if (dPosMag > 1e-9) {
        dir = dPos.unit();
    } else {
        dir = track->GetMomentumDirection();
    }
    gs.dirX = dir.x();
    gs.dirY = dir.y();
    gs.dirZ = dir.z();

    gs.charge = charge;
    gs.genType = MOP_GEN_CERENKOV;
    gs.materialId = GetMOPMaterialId(material);
    gs.numPhotons = (uint32_t)(numPhotons + 0.5);  // Fix 71: Geant4 호환 반올림
    gs.parentTrackId = track->GetTrackID();
    gs.betaInverse = betaInverse;
    gs.pmin = pmin / CLHEP::eV;  // C2 수정: Geant4 내부(MeV) → eV
    gs.pmax = pmax / CLHEP::eV;
    gs.maxCos = betaInverse / maxRI;
    gs.stepLength = stepLength / CLHEP::mm;

    // Fix 39: 스레드 로컬 벡터에 추가
    t_localGensteps.gensteps.push_back(gs);
    t_localGensteps.pendingPhotons += gs.numPhotons;
}

// ============================================================
// Primary opticalphoton (Beam source) 직접 캡처
// 호출자는 GPUOpticalKillProcess::PostStepDoIt (track 직접 전달)
// 또는 ProcessStep (step 전달, parentID/stepNumber 체크 후) 모두 지원.
// ============================================================
void TsGPUOpticalPhysics::CollectPrimaryPhoton(const G4Step* step,
                                               const G4Track* track) {
    if (!m_engine) return;

    if (!track && step) track = step->GetTrack();
    if (!track) return;

    const G4ThreeVector& pos = track->GetPosition();
    const G4ThreeVector& dir = track->GetMomentumDirection();
    G4ThreeVector        pol = track->GetPolarization();
    const G4Material*    mat = track->GetMaterial();

    // GPU-only fix (2026-04-25): TOPAS TsVGenerator.cc 의 random pol 버그
    // (line 368: fPolY = polarization.z()) 우회. invalid polarization
    // (not unit OR not perp to dir) 감지 시 GPU side 에서 valid random
    // pol 재생성. CPU side는 영향 없음 (Geant4 G4OpBoundaryProcess 그대로).
    // 정석 fix (2026-04-27): TOPAS TsVGenerator 가 BeamPolarization 을 BeamPos
    // RotY 회전 없이 G4 track 에 set 하는 경우, pol·dir != 0 (not transverse)
    // 가 됨. CPU G4OpBoundaryProcess 는 implicit 으로 parallel component 를
    // 제거하고 perpendicular projection 사용 → ppol 결과 correct.
    // GPU 도 동일 처리: parallel component 제거 + normalize.
    // 이전 random-transverse reset 은 ppol 을 unpolarized average 로 만들어
    // Brewster's angle 효과를 사라지게 했음 (U1 ppol bug 2026-04-27).
    {
        double polMag = pol.mag();
        double polDotDir = pol * dir;  // signed
        if (polMag < 0.99 || polMag > 1.01 || std::abs(polDotDir) > 0.01) {
            G4ThreeVector pol_perp = pol - polDotDir * dir;
            double perpMag = pol_perp.mag();
            if (perpMag > 1e-6) {
                // 정상: parallel component 제거 후 normalize → 올바른 transverse
                // axis 보존 (Fresnel 계산은 axis 만 사용, sign 무관).
                pol = pol_perp / perpMag;
            } else {
                // Degenerate: pol 이 dir 와 정확히 평행. 정의 불가 →
                // random transverse fallback.
                G4ThreeVector u_perp;
                if (std::abs(dir.x()) < 0.9)
                    u_perp = dir.cross(G4ThreeVector(1, 0, 0)).unit();
                else
                    u_perp = dir.cross(G4ThreeVector(0, 1, 0)).unit();
                G4ThreeVector v_perp = dir.cross(u_perp).unit();
                double phi = CLHEP::twopi * G4UniformRand();
                pol = std::cos(phi) * u_perp + std::sin(phi) * v_perp;
            }
        }
    }

    // Geant4 G4OpBoundaryProcess::PostStepDoIt() 라인 213-214 와 동일하게
    // track 이 보유한 polarization 을 그대로 사용. Geant4 는 input pol 을
    // 정규화하지 않고 받은 그대로 사용하지만, trivial face (fMaterial1 ==
    // fMaterial2) 에서 라인 456-462 의 SameMaterial early-exit 으로 Fresnel
    // 계산 자체를 skip → 비정규 polarization 의 영향이 nullified. 우리 GPU 는
    // shader 의 ProcessBoundaryFresnel/WithSurface 에서 동일한 early-exit 을
    // 구현하여 매칭 (matIn==matOut → pass-through).

    // MOPPhoton 필드 채우기 (Metal Photon struct과 layout 일치)
    MOPPhoton p{};
    p.posX = (float)(pos.x() / CLHEP::mm);
    p.posY = (float)(pos.y() / CLHEP::mm);
    p.posZ = (float)(pos.z() / CLHEP::mm);
    p.dirX = (float)dir.x();
    p.dirY = (float)dir.y();
    p.dirZ = (float)dir.z();
    p.polX = (float)pol.x();
    p.polY = (float)pol.y();
    p.polZ = (float)pol.z();

    // 광자 에너지 (eV)와 파장 (nm).
    double energy_eV = track->GetTotalEnergy() / CLHEP::eV;
    p.energy = (float)energy_eV;
    p.wavelength = (energy_eV > 0.0) ? (float)(1239.84193 / energy_eV) : 0.0f;
    p.time = (float)(track->GetGlobalTime() / CLHEP::ns);
    p.weight = (float)track->GetWeight();

    p.status = MOP_PHOTON_ALIVE;
    p.materialId = mat ? GetMOPMaterialId(mat) : 0u;
    // volumeId: m_volumeMap는 reverse (id->PV). Primary 발사 시 그냥 0으로 두고
    // GPU 측이 BVH traversal로 결정하도록 위임 — 첫 propagate step에서 갱신됨.
    p.volumeId = 0;
    p.stepCount = 0;
    p.flags = 0;

    // 2026-05-18: MT 안전 — engine 의 atomic counter 는 race-free 지만 storage
    // write 가 memory_order_relaxed (visibility 보장 X). caller 가 worker thread
    // 일 수 있으므로 m_gpuMutex 안에서 push (FlushBatch dispatch 와 직렬화).
    {
        std::lock_guard<std::recursive_mutex> lock(m_gpuMutex);
        MOPEngine_AddPrimaryPhoton(m_engine, &p);
    }

    static thread_local uint32_t pCount = 0;
    if (++pCount <= 200 && m_logLevel >= 2) {
        std::cout << "[TsGPU] Captured primary opticalphoton #" << pCount
                  << " pos=(" << p.posX << "," << p.posY << "," << p.posZ << ")mm"
                  << " dir=(" << p.dirX << "," << p.dirY << "," << p.dirZ << ")"
                  << " pol=(" << p.polX << "," << p.polY << "," << p.polZ << ")"
                  << " |pol|=" << std::sqrt(p.polX*p.polX + p.polY*p.polY + p.polZ*p.polZ)
                  << " dir·pol=" << (p.dirX*p.polX + p.dirY*p.polY + p.dirZ*p.polZ)
                  << " E=" << p.energy << "eV λ=" << p.wavelength << "nm"
                  << " matId=" << p.materialId << std::endl;
    }
}

// ============================================================
// 히트 및 통계
// ============================================================
uint32_t TsGPUOpticalPhysics::GetHits(MOPHit* hits, uint32_t maxHits) {
    if (!m_engine) return 0;
    return MOPEngine_GetHits(m_engine, hits, maxHits);
}

void TsGPUOpticalPhysics::GetStats(MOPPropagationStats* stats) {
    if (m_engine && stats) {
        MOPEngine_GetStats(m_engine, stats);
    }
}

void TsGPUOpticalPhysics::SetLogLevel(int level) {
    m_logLevel = level;
    if (m_engine) MOPEngine_SetLogLevel(m_engine, level);
}

void TsGPUOpticalPhysics::SetBaseSeed(uint32_t seed) {
    if (m_engine) MOPEngine_SetBaseSeed(m_engine, seed);
}

// Fix 41b: 스코어링 AABB 설정
// Multi-scorer race fix (2026-05-01): pending Propagate cmdBuf 가 AABB GPU buffer 를
// 읽는 중 CPU 가 덮어쓰면 race → heap corruption. m_gpuMutex 잠그고 pending
// 작업 harvest 후 안전하게 modify.
void TsGPUOpticalPhysics::SetScoringAABB(float minX, float minY, float minZ,
                                          float maxX, float maxY, float maxZ) {
    if (!m_engine) return;
    std::lock_guard<std::recursive_mutex> lock(m_gpuMutex);
    if (m_gpuPipelinePending) {
        HarvestPendingGPUResults();
    }
    MOPEngine_SetScoringAABB(m_engine, minX, minY, minZ, maxX, maxY, maxZ);
}

// Multi-scorer race fix (2026-05-01): public sync wrapper.
void TsGPUOpticalPhysics::WaitForPendingGPU() {
    if (!m_engine) return;
    std::lock_guard<std::recursive_mutex> lock(m_gpuMutex);
    if (m_gpuPipelinePending) {
        HarvestPendingGPUResults();
    }
}

// Multi-scorer race fix (2026-05-01): scorer DDA dispatch 를 m_gpuMutex 안에서 호출.
// 동시간에 worker thread 가 EndOfEvent 로 m_gpuMutex 잡고 Propagate dispatch 하면
// 같은 cmdQueue 에 두 cmdBuf 가 들어가 transit hit buffer / config buffer race.
// 1) pending Propagate harvest, 2) DDA dispatch, 3) DDA cmdBuf 동기 완료 대기.
void TsGPUOpticalPhysics::RunGPUDDAWithMaterialLocked(
    float compTransX, float compTransY, float compTransZ,
    float compFullX, float compFullY, float compFullZ,
    uint32_t nBinsX, uint32_t nBinsY, uint32_t nBinsZ,
    uint32_t scoringMaterialId)
{
    if (!m_engine) return;
    std::lock_guard<std::recursive_mutex> lock(m_gpuMutex);
    if (m_gpuPipelinePending) {
        HarvestPendingGPUResults();
    }
    MOPEngine_RunGPUDDA_WithMaterial(m_engine,
                                      compTransX, compTransY, compTransZ,
                                      compFullX, compFullY, compFullZ,
                                      nBinsX, nBinsY, nBinsZ,
                                      scoringMaterialId);
    // GetDDABinBuffer 는 내부적으로 pending DDA cmdBuf 동기 wait — read 용이지만 부수효과로 fence.
    (void)MOPEngine_GetDDABinBuffer(m_engine);
}

// Multi-scorer race fix (2026-05-01) — per-scorer accumulating buffer wrappers.
int TsGPUOpticalPhysics::RegisterScorerBinBufferLocked(uint32_t totalBins, uint32_t nBinsE,
                                                       float eMinEv, float eMaxEv) {
    if (!m_engine) return -1;
    std::lock_guard<std::recursive_mutex> lock(m_gpuMutex);
    if (m_gpuPipelinePending) {
        HarvestPendingGPUResults();
    }
    return MOPEngine_RegisterScorerBinBuffer(m_engine, totalBins, nBinsE, eMinEv, eMaxEv);
}

void TsGPUOpticalPhysics::SetScorerEnergyFilterLocked(int handle, float eLow, float eHigh) {
    if (!m_engine) return;
    std::lock_guard<std::recursive_mutex> lock(m_gpuMutex);
    MOPEngine_SetScorerEnergyFilter(m_engine, handle, eLow, eHigh);
}

void TsGPUOpticalPhysics::RunGPUDDAAccumulateLocked(
    int scorerHandle,
    float compTransX, float compTransY, float compTransZ,
    float compFullX, float compFullY, float compFullZ,
    uint32_t nBinsX, uint32_t nBinsY, uint32_t nBinsZ,
    uint32_t scoringMaterialId)
{
    if (!m_engine || scorerHandle < 0) return;
    std::lock_guard<std::recursive_mutex> lock(m_gpuMutex);
    // 2026-05-12 revert: inner Harvest 제거 시도가 torus_diffuse low-transit-rate
    // case (0.04% transit rate) 에서 역효과 (0.232). count mismatch 로 stale
    // buffer 영역 읽기. 일단 복원.
    if (m_gpuPipelinePending) {
        HarvestPendingGPUResults();
    }
    MOPEngine_RunGPUDDA_Accumulate(m_engine, scorerHandle,
                                    compTransX, compTransY, compTransZ,
                                    compFullX, compFullY, compFullZ,
                                    nBinsX, nBinsY, nBinsZ,
                                    scoringMaterialId);
    // 동기 wait 없음 — 누적 buffer 라 다음 dispatch 가 같은 cmdQueue FIFO 로 기다림.
    // Read 는 EndOfRun 에 GetScorerBinBufferLocked 가 처리.
}

void TsGPUOpticalPhysics::RunGPUDDAAccumulateExtLocked(
    int scorerHandle,
    float compTransX, float compTransY, float compTransZ,
    float compFullX, float compFullY, float compFullZ,
    uint32_t nBinsX, uint32_t nBinsY, uint32_t nBinsZ,
    uint32_t scoringMaterialId,
    uint32_t voxelType,
    float rMin, float phiStart, float thetaStart)
{
    if (!m_engine || scorerHandle < 0) return;
    std::lock_guard<std::recursive_mutex> lock(m_gpuMutex);
    if (m_gpuPipelinePending) {
        HarvestPendingGPUResults();
    }
    MOPEngine_RunGPUDDA_AccumulateExt(m_engine, scorerHandle,
                                       compTransX, compTransY, compTransZ,
                                       compFullX, compFullY, compFullZ,
                                       nBinsX, nBinsY, nBinsZ,
                                       scoringMaterialId,
                                       voxelType, rMin, phiStart, thetaStart);
}

const float* TsGPUOpticalPhysics::GetScorerBinBufferLocked(int scorerHandle,
                                                              uint32_t* outCount) {
    if (!m_engine || scorerHandle < 0) {
        if (outCount) *outCount = 0;
        return nullptr;
    }
    std::lock_guard<std::recursive_mutex> lock(m_gpuMutex);
    if (m_gpuPipelinePending) {
        HarvestPendingGPUResults();
    }
    return MOPEngine_GetScorerBinBuffer(m_engine, scorerHandle, outCount);
}

// SF1 (2026-04-24): GPU surface flux scorer 등록 wrapper
uint32_t TsGPUOpticalPhysics::RegisterSurfaceScorer(const MOPSurfaceDef& def) {
    if (!m_engine) return 0xFFFFFFFFu;
    return MOPEngine_RegisterSurfaceScorer(m_engine, &def);
}

void TsGPUOpticalPhysics::ClearSurfaceScorers() {
    if (m_engine) MOPEngine_ClearSurfaceScorers(m_engine);
}

// Fix 41: 볼륨 이름으로 volumeId 조회 (스코어러에서 사용)
int TsGPUOpticalPhysics::FindVolumeIdByName(const std::string& volumeName) const {
    for (const auto& pair : m_volumeMap) {
        if (pair.second && pair.second->GetName() == volumeName) {
            return (int)pair.first;
        }
    }
    // 빈닝 볼륨 이름(_NxNxN 접미사) 제거 후 재검색
    for (const auto& pair : m_volumeMap) {
        if (!pair.second) continue;
        std::string pvName = pair.second->GetName();
        // 볼륨 이름이 volumeName으로 시작하는지 확인
        if (pvName.find(volumeName) == 0) {
            return (int)pair.first;
        }
    }
    return -1;
}

void TsGPUOpticalPhysics::SetConfig(const MOPSimConfig& config) {
    if (!m_engine) return;
    MOPEngine_SetConfig(m_engine, &config);
}

MOPSimConfig TsGPUOpticalPhysics::GetConfig() const {
    MOPSimConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    if (m_engine) MOPEngine_GetConfig(m_engine, &cfg);
    return cfg;
}
