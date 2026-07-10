/**
 * TsGPUOpticalPhysicsModule.hh
 * TOPAS Physics Module Extension: Metal GPU 가속 광학 광자 전파
 *
 * G4VPhysicsConstructor 래퍼로, TOPAS Extension 시스템에 등록됨
 * ConstructProcess()에서 광학 광자 트랙을 kill하고 GPU로 전파를 오프로드
 *
 * TOPAS 파라미터 파일에서 사용:
 *   sv:Ph/Default/Modules = 3 "g4em-standard_opt4" "g4optical" "gpuoptical"
 */

#ifndef TS_GPU_OPTICAL_PHYSICS_MODULE_HH
#define TS_GPU_OPTICAL_PHYSICS_MODULE_HH

#include "G4VPhysicsConstructor.hh"
#include "G4OpticalPhoton.hh"
#include "G4VProcess.hh"
#include "G4VDiscreteProcess.hh"
#include "G4ProcessManager.hh"
#include "G4Step.hh"
#include "G4Track.hh"

// Forward declarations
class TsParameterManager;
class TsGPUOpticalPhysics;

/**
 * GPU 광학 광자 킬 프로세스
 * 광학 광자가 생성되면 즉시 kill하여 CPU 추적을 중단
 * 대신 GPU 엔진이 genstep 기반으로 광자를 전파
 */
class TsGPUOpticalKillProcess : public G4VProcess {
public:
    TsGPUOpticalKillProcess(const G4String& name = "GPUOpticalKill",
                             G4bool validationMode = false)
        : G4VProcess(name, fUserDefined),
          fValidationMode(validationMode), fEngine(nullptr) {}

    ~TsGPUOpticalKillProcess() override = default;

    void SetValidationMode(G4bool val) { fValidationMode = val; }
    G4bool GetValidationMode() const { return fValidationMode; }

    // GPU 엔진 핸들 주입 — primary opticalphoton (Beam source) 직접 캡처용.
    // 비-NULL이면 PostStepDoIt에서 primary 첫 step 광자를 GPU buffer로 push.
    void SetEngine(TsGPUOpticalPhysics* engine) { fEngine = engine; }

    // 이 프로세스는 PostStep에서만 동작
    G4double PostStepGetPhysicalInteractionLength(
        const G4Track&, G4double, G4ForceCondition* condition) override {
        if (fValidationMode) {
            // ValidationMode: 이 프로세스가 스텝을 제한하지 않음
            // g4optical이 정상적으로 광자를 추적하도록 함
            *condition = NotForced;
            return DBL_MAX;
        }
        *condition = StronglyForced;  // 항상 실행
        return 0.0;  // 즉시 실행 (kill)
    }

    // Definition은 .cc로 옮김 (TsGPUOpticalPhysics 헤더 의존성 회피)
    G4VParticleChange* PostStepDoIt(const G4Track& track, const G4Step& step) override;

private:
    G4bool fValidationMode;
    TsGPUOpticalPhysics* fEngine;  // primary photon capture target (nullable)

    // AlongStep / AtRest 는 사용 안 함
    G4double AlongStepGetPhysicalInteractionLength(
        const G4Track&, G4double, G4double, G4double&, G4GPILSelection*) override {
        return -1.0;
    }
    G4VParticleChange* AlongStepDoIt(const G4Track& track, const G4Step&) override {
        aParticleChange.Initialize(track);
        return &aParticleChange;
    }
    G4double AtRestGetPhysicalInteractionLength(const G4Track&, G4ForceCondition*) override {
        return -1.0;
    }
    G4VParticleChange* AtRestDoIt(const G4Track& track, const G4Step&) override {
        aParticleChange.Initialize(track);
        return &aParticleChange;
    }
};

/**
 * Genstep 수집 프로세스
 * 하전 입자의 매 스텝에서 호출되어 신틸레이션/체렌코프 genstep을 수집
 */
class TsGPUGenstepCollectorProcess : public G4VProcess {
public:
    TsGPUGenstepCollectorProcess(TsGPUOpticalPhysics* engine,
                                  const G4String& name = "GPUGenstepCollector")
        : G4VProcess(name, fUserDefined), fEngine(engine) {}

    ~TsGPUGenstepCollectorProcess() override = default;

    G4double PostStepGetPhysicalInteractionLength(
        const G4Track&, G4double, G4ForceCondition* condition) override {
        *condition = StronglyForced;
        return DBL_MAX;
    }

    G4VParticleChange* PostStepDoIt(const G4Track& track, const G4Step& step) override;

    G4double AlongStepGetPhysicalInteractionLength(
        const G4Track&, G4double, G4double, G4double&, G4GPILSelection*) override {
        return -1.0;
    }
    G4VParticleChange* AlongStepDoIt(const G4Track& track, const G4Step&) override {
        aParticleChange.Initialize(track);
        return &aParticleChange;
    }
    G4double AtRestGetPhysicalInteractionLength(const G4Track&, G4ForceCondition*) override {
        return -1.0;
    }
    G4VParticleChange* AtRestDoIt(const G4Track& track, const G4Step&) override {
        aParticleChange.Initialize(track);
        return &aParticleChange;
    }

private:
    TsGPUOpticalPhysics* fEngine;
};

/**
 * TOPAS Physics Module: GPU Optical Photon Physics
 * G4VPhysicsConstructor 서브클래스
 */
class TsGPUOpticalPhysicsModule : public G4VPhysicsConstructor {
public:
    TsGPUOpticalPhysicsModule(TsParameterManager* pM);
    ~TsGPUOpticalPhysicsModule() override;

    void ConstructParticle() override;
    void ConstructProcess() override;

private:
    TsParameterManager* fPm;
    TsGPUOpticalPhysics* fGPUEngine;
    TsGPUOpticalKillProcess* fKillProcess;
    TsGPUGenstepCollectorProcess* fCollectorProcess;
};

#endif /* TS_GPU_OPTICAL_PHYSICS_MODULE_HH */
