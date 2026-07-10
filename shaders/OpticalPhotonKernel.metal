/**
 * OpticalPhotonKernel.metal
 * 광학 광자 전파 메인 커널
 *
 * 물리 프로세스 경쟁 모델 (Geant4 호환):
 *   각 프로세스가 독립적으로 상호작용 길이를 샘플링 → 최단 거리 프로세스가 승리
 *   프로세스: 흡수, 레일리 산란, 미 산란, WLS 흡수, 경계 교차
 *
 * 경계 처리: Fresnel (s/p 편광 분해) + UNIFIED 표면 모델
 */

#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace metal::raytracing;

#include "Common.h"

// 진단 토글 (production: 0). build flag -DENABLE_DIAGNOSTICS=1 로 활성화.
#ifndef ENABLE_DIAGNOSTICS
#define ENABLE_DIAGNOSTICS 0
#endif

// regression 측정 전용 토글 (production: 0). build flag -DDISABLE_MESH_SIBLING_TEST=1
// 로 mesh tri runtime sibling adjacency 를 끄고 fix 전 동작을 재현한다.
#ifndef DISABLE_MESH_SIBLING_TEST
#define DISABLE_MESH_SIBLING_TEST 0
#endif

// Fix 62b: TIR 후 Geant4 Navigator push 모사 (Glass +X corner stuck 방지, production)
// 2026-06-12 FIX: 0 으로 비활성. Fix 62b 의 "TIR 시 photon.materialId = matIdOut 플립"은
// G4 와 발산 (G4OpBoundaryProcess.cc:1156-1177 은 TIR 에서 물질 불변) — TIR 광자는 유리
// 안에 남는데 Air 로 라벨되어 다음 facet 서 n1/n2 반전·공기경로에 BK7 흡수 적용·silent
// tunneling → 렌즈 marginal zone(내부출사각>임계각) 광자 오염 = lens-only −1.33%±0.37%
// (3시드)·r12-15 shell −2.92% 의 root cause. A/B 실측(-D...=0 metallib 교체):
// lens-only −0.642%→+0.695%, r12-15 −2.92%→+0.06%, core 불변. 원동기였던 "Glass +X
// corner stuck"은 2026-06-05 법선-only ε-push 통일이 대체 방지 (재발 시 정석 수리는
// push/dedup 측이지 material 플립 아님).
#define EMULATE_G4_TIR_PUSH 0

// ============================================================
// Fix 78b: function_constant 컴파일 타임 분기 제거
// 각 물리 프로세스 활성화 여부를 function_constant로 선언하면
// 컴파일러가 dead branch를 제거 → 레지스터/ALU 절약
// 엔진 측에서 newComputePipelineStateWithFunction:constantValues: 로 바인딩
// ============================================================
constant bool kEnableAbsorption [[function_constant(0)]];
constant bool kEnableRayleigh   [[function_constant(1)]];
constant bool kEnableMie        [[function_constant(2)]];
constant bool kEnableWLS        [[function_constant(3)]];
constant bool kEnableBoundary   [[function_constant(4)]];

// ============================================================
// 레일리 산란 — Geant4 G4OpRayleigh::PostStepDoIt line-by-line 재구현
// (정석 fix 2026-04-26: polarization-dependent angular distribution)
//
// Geant4 algorithm (G4OpRayleigh.cc:125-170):
//   do {
//     cost = G4UniformRand();          // uniform [0,1]
//     sint = sqrt(1 - cost²);
//     if (G4UniformRand() < 0.5) cost = -cost;  // 50% sign flip → uniform [-1,1]
//     phi = twopi * G4UniformRand();
//     newMomDir.set(sint·cosphi, sint·sinphi, cost);
//     newMomDir.rotateUz(oldMomDir);   // local frame = oldMomDir (axis = beam dir)
//     oldPol = particle->GetPolarization();
//     newPol = (oldPol - newMomDir·oldPol × newMomDir).unit();  // project to ⊥ plane
//     if (newPol.mag() == 0) { random azimuthal pol }
//     else { 50% sign flip newPol }
//     cosTheta = newPol·oldPol;
//   } while (cosTheta² < G4UniformRand());   // ← polarization-based rejection
//
// 이전 GPU 구현은 (1+cos²θ)/2 direction-based sampling 으로 polarization
// dependence 누락 → CPU/GPU 10%+ fluence 차이 (U3v1side 검증).
inline void DoRayleighScattering(thread Photon& photon, thread RandomState& rng) {
    float3 oldDir = float3(photon.direction);
    float3 oldPol = float3(photon.polarization);

    // 정석 fix (2026-04-26): pre-loop oldPol 보정 제거.
    // Geant4 G4OpRayleigh::PostStepDoIt 와 Opticks qsim::rayleigh_scatter 모두
    // pre-loop polarization 사전 normalize/체크 없음. oldPol을 그대로 사용하고
    // corner case (newPol projection magnitude == 0) 만 do-while loop 안의
    // polLen < 1e-6 branch 에서 처리. 사전 보정 시 Geant4와 다른 random pol
    // 사용으로 statistical bias 가능 (U3 4.1% sum diff 후보).

    float3 newMomDir;
    float3 newPol;
    const int maxRayleighRetries = 32;
    for (int retry = 0; retry < maxRayleighRetries; retry++) {
        // Step 1: cost uniform [0, 1] + 50% sign flip → uniform [-1, 1]
        float cost = RandUniform(rng);
        float sint = MOP_SQRT(max(0.0f, 1.0f - cost * cost));
        if (RandUniform(rng) < 0.5f) cost = -cost;

        // Step 2: phi uniform [0, 2π]
        float phi = TWO_PI * RandUniform(rng);
        float cosphi = MOP_COS(phi);
        float sinphi = MOP_SIN(phi);

        // Step 3: newMomDir in oldMomDir local frame, then rotateUz → world frame
        // rotateUz(u): rotate (vx,vy,vz) so that local +z aligns with u.
        // Geant4 G4ThreeVector::rotateUz formula:
        //   u_perp = sqrt(u.x² + u.y²)
        //   if u_perp > 0:
        //     px = (u.x·u.z·vx - u.y·vy)/u_perp + u.x·vz
        //     py = (u.y·u.z·vx + u.x·vy)/u_perp + u.y·vz
        //     pz = -u_perp·vx + u.z·vz
        //   else (u parallel to ±z): px=±vx, py=vy, pz=±vz
        float vx = sint * cosphi;
        float vy = sint * sinphi;
        float vz = cost;
        float ux = oldDir.x, uy = oldDir.y, uz = oldDir.z;
        float u_perp2 = ux*ux + uy*uy;
        if (u_perp2 > 1e-12f) {
            float u_perp = MOP_SQRT(u_perp2);
            newMomDir.x = (ux * uz * vx - uy * vy) / u_perp + ux * vz;
            newMomDir.y = (uy * uz * vx + ux * vy) / u_perp + uy * vz;
            newMomDir.z = -u_perp * vx + uz * vz;
        } else {
            // oldDir 이 ±z 축에 정렬: 단순 sign 적용
            newMomDir.x = (uz > 0.0f) ? vx : -vx;
            newMomDir.y = vy;
            newMomDir.z = (uz > 0.0f) ? vz : -vz;
        }
        newMomDir = normalize(newMomDir);

        // Step 4: newPol = (oldPol - (newMomDir·oldPol) × newMomDir).unit()
        // (oldPol 을 newMomDir 에 수직인 평면으로 project)
        float3 polProj = oldPol - dot(newMomDir, oldPol) * newMomDir;
        float polLen = length(polProj);
        if (polLen < 1e-6f) {
            // Step 5a: corner case — oldPol 이 newMomDir 과 평행
            // random azimuthal 새 polarization
            float pphi = RandUniform(rng) * TWO_PI;
            float3 raw = float3(MOP_COS(pphi), MOP_SIN(pphi), 0.0f);
            // rotateUz to newMomDir frame
            float nx = newMomDir.x, ny = newMomDir.y, nz = newMomDir.z;
            float n_perp2 = nx*nx + ny*ny;
            if (n_perp2 > 1e-12f) {
                float n_perp = MOP_SQRT(n_perp2);
                newPol.x = (nx * nz * raw.x - ny * raw.y) / n_perp + nx * raw.z;
                newPol.y = (ny * nz * raw.x + nx * raw.y) / n_perp + ny * raw.z;
                newPol.z = -n_perp * raw.x + nz * raw.z;
            } else {
                newPol.x = (nz > 0.0f) ? raw.x : -raw.x;
                newPol.y = raw.y;
                newPol.z = (nz > 0.0f) ? raw.z : -raw.z;
            }
            newPol = normalize(newPol);
        } else {
            // Step 5b: normal case — polProj normalize, 50% sign flip
            newPol = polProj / polLen;
            if (RandUniform(rng) < 0.5f) newPol = -newPol;
        }

        // Step 6: polarization-based rejection — cos²θ_pol vs rand
        float cosThetaPol = dot(newPol, oldPol);
        if (cosThetaPol * cosThetaPol >= RandUniform(rng)) {
            // accept
            photon.direction = packed_float3(newMomDir);
            photon.polarization = packed_float3(newPol);
            return;
        }
        // else: 다음 retry
    }

    // safety: retry 한도 도달 시 마지막 sample 사용 (rejection 미통과)
    photon.direction = packed_float3(newMomDir);
    photon.polarization = packed_float3(newPol);
}

// ============================================================
// G4 ThreeVector::rotateUz — local frame vector 를 newUz axis frame 으로 회전.
// CLHEP/Geometry/Vector/ThreeVector.h 의 rotateUz 정석 정합.
// 2026-05-17: fp32 intermediate round 회피 위해 multi-line variable 분리 대신 단일
// expression 으로 작성 → Metal compiler FMA contraction 적극 적용 (fp32 ULP 절반 감소).
// G4 의 (local x,y,z) = (sinθ cosφ, sinθ sinφ, cosθ) → world rotation.
// ============================================================
inline float3 rotateUz_G4(float3 local, float3 newUz) {
    // up² = u1² + u2² (axis 의 transverse magnitude²)
    if (newUz.x*newUz.x + newUz.y*newUz.y > 1e-20f) {
        // FMA-friendly single-expression — Metal `fma()` 명시 사용으로 intermediate
        // round 강제 회피. 분리된 variable + `/` 보다 `precise::divide` 또는 FMA-mul
        // sequence 가 더 정확.
        return float3(
            fma(newUz.x*newUz.z*local.x - newUz.y*local.y,
                metal::rsqrt(newUz.x*newUz.x + newUz.y*newUz.y),
                newUz.x*local.z),
            fma(newUz.y*newUz.z*local.x + newUz.x*local.y,
                metal::rsqrt(newUz.x*newUz.x + newUz.y*newUz.y),
                newUz.y*local.z),
            fma(-MOP_SQRT(newUz.x*newUz.x + newUz.y*newUz.y), local.x,
                newUz.z*local.z)
        );
    } else if (newUz.z < 0.0f) {
        return float3(-local.x, local.y, -local.z);
    }
    return local;
}

// ============================================================
// 미 산란 (Henyey-Greenstein) 방향 + 편광 업데이트
// ============================================================
inline void DoMieScattering(thread Photon& photon, thread RandomState& rng,
                             float gForward, float gBackward, float forwardRatio) {
    // G4OpMieHG.cc:96-124 정석 매칭:
    //   forward branch (ξ ≤ ratio): g = MIEHG_FORWARD,  direction = +1
    //   else branch:                 g = MIEHG_BACKWARD, direction = -1
    //   HG inverse CDF 로 θ sample (forward-peaked, g 부호로 결정),
    //   direction == -1 시 θ → π - θ (즉 cosθ → -cosθ) flip 해서 backward-peaked.
    //
    // 2026-05-16 bug fix: 이전 코드는 G4 의 cos flip 누락. 사용자가 G4 관행
    // 대로 MIEHG_BACKWARD = +0.9 공급 시 GPU 는 forward 산란만 수행 (잘못).
    // 검증: test_run/_mie_backward_* CPU back/front = 0.253/0.020,
    //       GPU (fix 전) = 0.168/0.102 (= forward 결과 bit-identical).
    float g;
    int direction;
    if (RandUniform(rng) <= forwardRatio) {
        g = gForward;
        direction = 1;
    } else {
        g = gBackward;
        direction = -1;
    }

    float cost;
    if (abs(g) < 1e-6) {
        cost = 2.0 * RandUniform(rng) - 1.0;
    } else {
        float xi = RandUniform(rng);
        float tmp = (1.0 - g * g) / (1.0 - g + 2.0 * g * xi);
        cost = (1.0 + g * g - tmp * tmp) / (2.0 * g);
    }

    // G4 line 123-124: backward 시 cos flip.
    if (direction == -1) {
        cost = -cost;
    }

    // 2026-05-17: fp32 catastrophic cancellation 회피 —
    //   sint = sqrt(1-cost²) 는 cost≈±1 시 (1-cost)(1+cost) cancellation.
    //   factored form (1-cost)*(1+cost) 사용 시 ULP loss 절반 감소.
    // newDir 계산을 single expression 으로 작성 → Metal FMA contraction 적극.
    float sint = MOP_SQRT(max(0.0f, (1.0f - cost) * (1.0f + cost)));
    float phi = TWO_PI * RandUniform(rng);
    float3 dir = float3(photon.direction);
    float3 newDir = rotateUz_G4(float3(sint * MOP_COS(phi), sint * MOP_SIN(phi), cost), dir);

    // 편광 업데이트 — G4OpMieHG.cc:135-150 정석 매칭.
    //   newPol_G4 = (newMomDir - oldPol/(newMomDir·oldPol)).unit()
    //   GPU 의 newDir × (newDir × oldPol) = (newDir·oldPol)*newDir - oldPol
    //     = c * (newDir - oldPol/c)  (c = newDir·oldPol)
    //   즉 정규화 후 G4 와 동일 (up to sign), 그래서 polCross/|polCross| 채택.
    //
    // 2026-05-16 추가 fix:
    //   - (else 분기) G4 line 147-149: 50% 확률 newPol = -newPol (newDir 에 수직인
    //     두 방향 중 하나를 random 선택). 이전 GPU 는 deterministic sign.
    //   - (fallback) G4 line 139-144: polLen ≈ 0 시 newDir 에 수직인 random φ
    //     unit vector. (cos r, sin r, 0).rotateUz(newDir). 이전 GPU 는
    //     deterministic perpendicular.
    // Fluence (편광 무관) observable 엔 영향 없으나 편광 의존 (Fresnel 후 Mie
    // 또는 Mie 후 Fresnel) 셋업 정확성 위해 G4-정석.
    float3 oldPol = float3(photon.polarization);
    float3 polCross = cross(newDir, cross(newDir, oldPol));
    float polLen = length(polCross);
    float3 newPol;
    if (polLen > 1e-6) {
        newPol = polCross / polLen;
        // G4 line 147-149: random ±1 sign flip.
        if (RandUniform(rng) < 0.5) {
            newPol = -newPol;
        }
    } else {
        // G4 line 141-143: r = uniform * twopi; (cos r, sin r, 0).rotateUz(newDir)
        // GPU 구현: newDir 에 수직인 두 basis u/v 만든 뒤 cos r * u + sin r * v.
        float r = TWO_PI * RandUniform(rng);
        float3 upPol = abs(newDir.y) < 0.999 ? float3(0,1,0) : float3(1,0,0);
        float3 uPol = normalize(cross(upPol, newDir));
        float3 vPol = cross(newDir, uPol);
        newPol = MOP_COS(r) * uPol + MOP_SIN(r) * vPol;
    }

    photon.direction = packed_float3(newDir);
    photon.polarization = packed_float3(newPol);
}

// ============================================================
// Geant4 amplitude method 기반 boundary physics (G4OpBoundaryProcess.cc
// ApplyDielectricBoundaryTransition lines 1440-1580 와 동일).
//
// 입사 polarization E1 을 surface 평면 기준 perpendicular(s) 와
// parallel(p) 로 분해한 뒤 Fresnel transmitted amplitude (E2_perp,
// E2_parl) 를 계산. transCoeff = s2/s1 = Poynting flux ratio 로
// 반사/투과 결정. 반사 / 굴절된 polarization 도 amplitude 방식으로
// 회전시켜 multi-bounce Rs/Rp 분리 추적이 정확하게 수행되도록 함.
//
// FresnelDecompose() 는 한 번 분해하면 R 결정 + 양쪽 polarization
// update 모두 동일 데이터 (E1_perp, E1_parl, E2_perp, E2_parl, A_trans)
// 로 끝낼 수 있어 Geant4 와 동일한 코드 경로를 유지함.
// ============================================================
struct FresnelState {
    float E1_perp;       // 입사 s amplitude (scalar, signed)
    float E1_parl;       // 입사 p amplitude (scalar, ≥0)
    float E2_perp;       // 굴절 s amplitude (scalar)
    float E2_parl;       // 굴절 p amplitude (scalar)
    float E2_total;      // E2_perp² + E2_parl²
    float cosT;          // 굴절 cosθ_t (≥0)
    float n1;
    float n2;
    float3 A_trans;      // s 단위벡터 = (oldDir × normal).unit()
    bool   tir;          // 전반사 여부
    bool   normalIncidence; // sint1 ≈ 0
};

inline FresnelState FresnelDecompose(float cosI, float n1, float n2,
                                     float3 dir, float3 normal, float3 pol) {

    FresnelState st;
    st.n1 = n1;
    st.n2 = n2;
    st.tir = false;
    st.normalIncidence = false;
    st.E1_perp = 0.0;
    st.E1_parl = 0.0;
    st.E2_perp = 0.0;
    st.E2_parl = 0.0;
    st.E2_total = 0.0;
    st.cosT = 0.0;
    st.A_trans = float3(0);

    float3 crossDN = cross(dir, normal);
    float sint1_len = length(crossDN);
    if (sint1_len < 1e-6) {
        // 수직 입사 — Geant4 라인 1440-1476, 1557-1559, 1577-1580
        st.normalIncidence = true;
        st.A_trans = pol;
        st.E1_perp = 0.0;
        st.E1_parl = 1.0;
        // 수직 입사에서는 cosT = 1, sinT = 0
        float sinT2 = (n1 / n2) * (n1 / n2) * (1.0 - cosI * cosI);
        if (sinT2 >= 1.0) { st.tir = true; return st; }
        st.cosT = MOP_SQRT(1.0 - sinT2);
        // E2_perp/parl 는 amplitude method 로 계산 (수직 입사도 동일 식)
        float s1 = n1 * cosI;
        st.E2_perp = 2.0 * s1 * st.E1_perp / (n1 * cosI + n2 * st.cosT);
        st.E2_parl = 2.0 * s1 * st.E1_parl / (n2 * cosI + n1 * st.cosT);
        st.E2_total = st.E2_perp * st.E2_perp + st.E2_parl * st.E2_parl;
        return st;
    }

    // 편광 분해 (Geant4 라인 1480-1492)
    st.A_trans = crossDN / sint1_len;
    st.E1_perp = dot(pol, st.A_trans);
    float3 E1pp = st.E1_perp * st.A_trans;
    float3 E1pl = pol - E1pp;
    st.E1_parl = length(E1pl);

    // TIR 체크 — P2 정석 fix: catastrophic cancellation 회피 (1-cosI²) → (1-cosI)(1+cosI)
    // multi-bounce 시 cosI≈1 영역에서 ULP 손실 누적 방지.
    float ratio = n1 / n2;
    float ratio_cosI = ratio * cosI;
    // sinT² = ratio² - (ratio*cosI)² = (ratio - ratio*cosI)(ratio + ratio*cosI)
    float sinT2 = (ratio - ratio_cosI) * (ratio + ratio_cosI);
    if (sinT2 >= 1.0) { st.tir = true; return st; }

    st.cosT = MOP_SQRT(1.0 - sinT2);

    // Fresnel transmitted amplitudes (Geant4 라인 1519-1523)
    float s1 = n1 * cosI;
    st.E2_perp = 2.0 * s1 * st.E1_perp / (n1 * cosI + n2 * st.cosT);
    st.E2_parl = 2.0 * s1 * st.E1_parl / (n2 * cosI + n1 * st.cosT);
    st.E2_total = st.E2_perp * st.E2_perp + st.E2_parl * st.E2_parl;
    return st;
}

// 반사 확률 (transCoeff = s2/s1 → 1 - transCoeff = R)
// Geant4 라인 1505-1512.
inline float FresnelReflectance(float cosI, float n1, float n2,
                                 float3 dir, float3 normal, float3 pol,
                                 thread float& Rs, thread float& Rp) {
    FresnelState st = FresnelDecompose(cosI, n1, n2, dir, normal, pol);
    if (st.tir) { Rs = 1.0; Rp = 1.0; return 1.0; }

    // Rs, Rp 도 amplitude 로부터 (진단/외부 노출용)
    float ts = st.E2_perp / max(abs(st.E1_perp), 1e-12);
    float tp = st.E2_parl / max(abs(st.E1_parl), 1e-12);
    if (abs(st.E1_perp) < 1e-12) ts = 2.0 * st.n1 * cosI / (st.n1 * cosI + st.n2 * st.cosT);
    if (abs(st.E1_parl) < 1e-12) tp = 2.0 * st.n1 * cosI / (st.n2 * cosI + st.n1 * st.cosT);
    float Ts = (st.n2 * st.cosT) / (st.n1 * cosI) * ts * ts;
    float Tp = (st.n2 * st.cosT) / (st.n1 * cosI) * tp * tp;
    Rs = saturate(1.0 - Ts);
    Rp = saturate(1.0 - Tp);

    if (st.normalIncidence) return Rs;  // E1_perp=0, E1_parl=1 → 단일 channel

    float s1 = st.n1 * cosI;
    float s2 = st.n2 * st.cosT * st.E2_total;
    float trans = s2 / max(s1, 1e-12);
    return saturate(1.0 - trans);
}

// 반사 후 polarization (Geant4 라인 1547-1559)
inline float3 UpdatePolarizationReflect(float3 dir, float3 newDir, float3 pol, float3 normal,
                                         float n1, float n2) {
    float cosI = -dot(dir, normal);
    if (cosI < 0.0) { normal = -normal; cosI = -cosI; }
    FresnelState st = FresnelDecompose(cosI, n1, n2, dir, normal, pol);

    if (st.tir) {
        // Geant4 라인 1500: -oldPol + 2(oldPol·n)n
        float3 result = -pol + 2.0 * dot(pol, normal) * normal;
        float rLen = length(result);
        return (rLen > 1e-6) ? (result / rLen) : pol;
    }

    if (st.normalIncidence) {
        // Geant4 라인 1558: ±oldPolarization (n2>n1 → -)
        return (n2 > n1 ? -1.0 : 1.0) * pol;
    }

    // 정석 fix (G4 expert, Opticks 방식): reflected amplitude.
    // P-amplitude는 Opticks 형식 (분자 우선 곱셈) 으로 변경: n2*E2_parl/n1.
    float E2_parl_r = st.n2 * st.E2_parl / st.n1 - st.E1_parl;
    float E2_perp_r = st.E2_perp - st.E1_perp;

    float3 A_paral = cross(newDir, st.A_trans);
    float A_paral_len = length(A_paral);
    if (A_paral_len < 1e-6) return pol;
    A_paral /= A_paral_len;

    // Opticks 방식: 2D scalar normalize 후 unit-orthogonal basis 합성.
    // basis (A_trans, A_paral) 가 이미 단위 + 직교 → 결과도 단위 (이중 normalize 불필요).
    float2 RR = float2(E2_perp_r, E2_parl_r);
    float RR_len2 = dot(RR, RR);
    if (RR_len2 < 1e-20) return pol;
    RR /= MOP_SQRT(RR_len2);
    return RR.x * st.A_trans + RR.y * A_paral;
}

// 굴절 후 polarization (Geant4 라인 1568-1580)
inline float3 UpdatePolarizationRefract(float3 dir, float3 newDir, float3 pol, float3 normal,
                                         float n1, float n2) {
    float cosI = -dot(dir, normal);
    if (cosI < 0.0) { normal = -normal; cosI = -cosI; }
    FresnelState st = FresnelDecompose(cosI, n1, n2, dir, normal, pol);

    if (st.tir) return pol;
    if (st.normalIncidence) return pol;

    float3 A_paral = cross(newDir, st.A_trans);
    float A_paral_len = length(A_paral);
    if (A_paral_len < 1e-6) return pol;
    A_paral /= A_paral_len;

    // Opticks 방식: 2D normalize + basis 합성 (이중 normalize 제거).
    float2 TT = float2(st.E2_perp, st.E2_parl);
    float TT_len2 = dot(TT, TT);
    if (TT_len2 < 1e-20) return pol;
    TT /= MOP_SQRT(TT_len2);
    return TT.x * st.A_trans + TT.y * A_paral;
}

// reflect: newDir = dir - 2*dot(dir, normal)*normal (caller flips normal so cosI ≥ 0)
inline float3 reflect_direction(float3 dir, float3 normal) {
    float3 newDir = dir - 2.0f * dot(dir, normal) * normal;
    float n2 = dot(newDir, newDir);
    return (n2 > 1e-30f) ? newDir * rsqrt(n2) : newDir;
}

// refract: newDir = ratio*dir + (ratio*cosI - cosT)*normal
inline float3 refract_direction(float3 dir, float ratio, float cosI, float cosT, float3 normal) {
    float3 newDir = ratio * dir + (ratio * cosI - cosT) * normal;
    float n2 = dot(newDir, newDir);
    return (n2 > 1e-30f) ? newDir * rsqrt(n2) : newDir;
}

// ============================================================
// Fresnel 전용 경계 처리 (표면 속성 없는 경우)
// ============================================================
inline void ProcessBoundaryFresnel(
    thread Photon& photon,
    thread RandomState& rng,
    float3 normal,
    uint matIdOutParam,                // Fix 71: uint로 변경 — MaterialGPU 32KB 캐시라인 fetch 방지
    device const MaterialGPU* materials, // fallback InterpolateProperty용
    uint matIdInParam,
    device atomic_uint* diagCounters,    // 2026-05-09: lens TIR/refl/transmit count
    float n1_pre = 0.0,   // Fix 64b: 사전 계산된 n1 (0이면 내부 계산)
    float n2_pre = 0.0)
{
    // Geant4 G4OpBoundaryProcess.cc 라인 456-462 의 SameMaterial early-exit:
    // 두 측 material 이 동일하면 (matIn==matOut) Fresnel 계산을 skip 하고
    // photon 을 그대로 통과시킨다. AirHalfScorer-World 같이 같은 material 의
    // sub-volume 경계가 spurious R 을 만들지 않도록 하는 핵심 분기.
    // 2026-05-13: SameMaterial 시 ε push 추가 — self-intersection 차단.
    // 다른 boundary 분기 (Fresnel reflect/refract/TIR) 와 일관성. 부재 시 광자가
    // 같은 위치에서 다음 step BVH 시작 → 같은 또는 인접 triangle hit → loop.
    // Air-Air boundary (ImagingSensor-World 등) 만남마다 광자 lose. Air-only
    // world 시나리오 검증: 0.074 → 0.979 (90.5% recovery).
    // Push direction = dir 쪽으로 sign-aligned normal (광자 진행 방향과 같은
    // 쪽으로 normal 정렬해서 face 통과 보장).
    if (matIdInParam == matIdOutParam) {
        photon.materialId = matIdOutParam;
        const float kBoundaryPushEpsSame = 1e-4f;  // 100 nm — Fresnel ε push 와 동일
        float3 dir_ = float3(photon.direction);
        float dn = dot(dir_, normal);
        float3 pushN = (dn >= 0.0f) ? normal : -normal;
        photon.position = packed_float3(float3(photon.position) + kBoundaryPushEpsSame * pushN);
        return;
    }

    float n1 = (n1_pre > 0.0) ? n1_pre : InterpolateProperty(materials[matIdInParam].refractiveIndex, photon.energy);
    float n2 = (n2_pre > 0.0) ? n2_pre : InterpolateProperty(materials[matIdOutParam].refractiveIndex, photon.energy);
    if (n1 <= 0.0) n1 = 1.0;
    if (n2 <= 0.0) n2 = 1.0;

    float3 dir = float3(photon.direction);
    float3 pol = float3(photon.polarization);
    float cosI = -dot(dir, normal);
    if (cosI < 0.0) {
        normal = -normal;
        cosI = -cosI;
    }

    // 2026-05-20 H3 fix: amplitude method 의 fp32 stable form.
    // Polarization-별 R = |E_perp|² * R_s + |E_parl|² * R_p (G4 정석).
    // Ensemble mean 은 polarization-averaged 와 동일, single photon multi-bounce
    // 시 polarization correlation 의 systematic 정확 반영.
    // fp32 stable: |E_perp|² = dot²(pol, A_trans), |E_parl|² = 1 - |E_perp|² (unit pol).
    // R_s, R_p 의 sinT²/cosT 계산 도 cancellation 회피 form.
    float Rs = 0, Rp = 0;
    float R;
    {
        float3 crossDN_p = cross(dir, normal);
        float sint1_len_p = length(crossDN_p);
        float3 A_trans_p = (sint1_len_p > 1e-6f) ? (crossDN_p / sint1_len_p) : float3(1,0,0);
        float E_perp_amp = dot(pol, A_trans_p);
        float E_perp_sq = E_perp_amp * E_perp_amp;
        float E_parl_sq = saturate(1.0f - E_perp_sq);

        float ratio_p = n1 / n2;
        float ratio_cosI = ratio_p * cosI;
        float sinT2_p = (ratio_p - ratio_cosI) * (ratio_p + ratio_cosI);
        if (sinT2_p >= 1.0f) {
            R = 1.0f;  // TIR
            Rs = 1.0f; Rp = 1.0f;
        } else {
            float sinT_p = sqrt(max(sinT2_p, 0.0f));
            float cosT_p = sqrt((1.0f - sinT_p) * (1.0f + sinT_p));
            float n1cosI = n1 * cosI;
            float n2cosT = n2 * cosT_p;
            float n2cosI = n2 * cosI;
            float n1cosT = n1 * cosT_p;
            float Rs_a = (n1cosI - n2cosT) / (n1cosI + n2cosT);
            float Rp_a = (n2cosI - n1cosT) / (n2cosI + n1cosT);
            Rs = Rs_a * Rs_a;
            Rp = Rp_a * Rp_a;
            // 광자 polarization 별 R (amplitude method G4 정석)
            R = E_perp_sq * Rs + E_parl_sq * Rp;
        }
    }
    // Wächter-Binder offset_ray 정석 fix (L1+L2 expert, 2026-04-23 / 보강 2026-04-25):
    // 모든 boundary 처리 후 ε push → self-intersection 누설 차단.
    // 2026-04-25 보강: push 방향을 newDir 단독 → newDir + normal 합성으로 변경.
    //   이유: grazing 각도 (예: 85° 입사) 에서 newDir 의 normal 성분이 작아
    //         (cos85°=0.087) push 후 plane crossing 보장 안됨 → SF1 sign-change
    //         detection 미발화 → R 측정 누락 발생.
    //   수정: refract 시 -normal 방향 (새 매질 쪽), reflect/TIR 시 +normal 방향
    //         (원 매질 쪽) 으로도 확실하게 push → 어떤 grazing 각도에서도
    //         plane crossing/non-crossing 일관 보장.
    // ε = 1e-4mm (100nm) > Apple Metal float32 ULP at 100mm = 12nm.
    // (cosI<0 분기 후 normal 은 photon 의 "들어가는" 쪽으로 flip 되어 있음.
    //  refract: photon 이 -normal 쪽으로 진행 → -normal 방향 push.
    //  reflect: photon 이 +normal 쪽에 머묾 → +normal 방향 push.)
    const float kBoundaryPushEps = 1e-4f;  // production default
    if (RandUniform(rng) < R) {
        // 반사 — Fresnel reflection (CPU TOPAS FresnelReflection 또는 TIR)
        photon.reflectedCount++;
        #if ENABLE_DIAGNOSTICS
        atomic_fetch_add_explicit(&diagCounters[178], 1u, memory_order_relaxed);  // lens Fresnel reflect (orig)
        #endif
        // S1 엄밀: dd_t evaluation, fp32 cast at storage
        float3 newDir = reflect_direction(dir, normal);
        float3 newPol = UpdatePolarizationReflect(dir, newDir, pol, normal, n1, n2);
        photon.direction = packed_float3(newDir);
        photon.polarization = packed_float3(newPol);
        // Reflect: 원 매질 쪽 (+normal 방향) 으로 push.  [A/B: 법선-only]
        photon.position = packed_float3(float3(photon.position)
            + (2.0f * kBoundaryPushEps) * normal);
    } else {
        // 굴절 (Snell's law) — P2 정석 fix: cancellation 회피 reformulation
        float ratio = n1 / n2;
        float ratio_cosI = ratio * cosI;
        float sinT2 = (ratio - ratio_cosI) * (ratio + ratio_cosI);

        if (sinT2 > 1.0) {
            // 전반사 (안전장치) — TotalInternalReflection
            photon.reflectedCount++;
            #if ENABLE_DIAGNOSTICS
            atomic_fetch_add_explicit(&diagCounters[179], 1u, memory_order_relaxed);  // lens TIR (orig)
            #endif
            // S1 엄밀: dd_t reflect
            float3 newDir = reflect_direction(dir, normal);
            float3 newPol = UpdatePolarizationReflect(dir, newDir, pol, normal, n1, n2);
            photon.direction = packed_float3(newDir);
            photon.polarization = packed_float3(newPol);
            // TIR: 원 매질 쪽 (+normal 방향) 으로 push.  [A/B: 법선-only]
            photon.position = packed_float3(float3(photon.position)
                + (2.0f * kBoundaryPushEps) * normal);
        } else {
            #if ENABLE_DIAGNOSTICS
            atomic_fetch_add_explicit(&diagCounters[181], 1u, memory_order_relaxed);  // lens transmit (orig)
            #endif
            float cosT = MOP_SQRT(1.0 - sinT2);
            // S1 엄밀: dd_t refract
            float3 newDir = refract_direction(dir, ratio, cosI, cosT, normal);
            float3 newPol = UpdatePolarizationRefract(dir, newDir, pol, normal, n1, n2);
            photon.direction = packed_float3(newDir);
            photon.polarization = packed_float3(newPol);
            photon.materialId = matIdOutParam;  // Fix 71: uint 직접 사용
            // Refract: 새 매질 쪽 (-normal 방향) 으로 push (face 통과 보장).  [A/B: 법선-only]
            photon.position = packed_float3(float3(photon.position)
                - (2.0f * kBoundaryPushEps) * normal);
        }
    }
}

// ============================================================
// Fix 2026-05-16 (G4 정석 매칭): Geant4 UNIFIED microfacet α 샘플링.
// G4OpBoundaryProcess::GetFacetNormal (line 683-692 of G4OpBoundaryProcess.cc):
//   f_max = min(1.0, 4 * sigma_alpha)
//   do {
//     alpha = G4RandGauss::shoot(0.0, sigma_alpha)   // FULL Gauss (음수 가능)
//     sinAlpha = sin(alpha)
//   } while (uniform * f_max > sinAlpha  ||  alpha >= halfpi)
//
// 이전 GPU 구현 (RandHalfGaussian + accept if uniform < sin(α)):
//   1) f_max scaling 누락 → saturated regime (α > 0.41 rad at σα=0.1) 에서 차이
//   2) HalfGauss (|·|) → 음수 α 의 자연스러운 rejection 흐름 못 따라감
//   둘 다 distribution shape 영향.
//
// 본 fix: G4 와 동일하게 full Gauss + f_max scaling + 음수 α 도 sin(α)<0 으로
//        자동 거부. sigmaAlpha=0 시 α=0 fast path 유지.
inline float SampleMicrofacetAlpha(thread RandomState& rng, float sigmaAlpha) {
    if (sigmaAlpha <= 0.0) return 0.0;
    const float f_max = min(1.0f, 4.0f * sigmaAlpha);
    const int maxIter = 100;  // G4 도 무한 루프 가능, 안전 cap
    for (int i = 0; i < maxIter; i++) {
        float alpha = RandGauss(rng, sigmaAlpha);  // full Gauss
        float sinA = MOP_SIN(alpha);
        // G4: while (uniform*f_max > sinA || alpha >= halfpi) repeat
        //  ↔ accept = (uniform*f_max <= sinA) && (alpha < halfpi)
        if (alpha < PI * 0.5f && RandUniform(rng) * f_max <= sinA) return alpha;
    }
    return min(sigmaAlpha, PI * 0.5f - 0.001f);  // fallback (rare)
}

// Fix 2026-05-16: G4 GetFacetNormal 의 do-while 정석 매칭 — facet 이 photon 향해
// 안 하면 reject + 재샘플 (NOT flip). G4 line 686-698:
//   do { alpha = sample(); ... } while (momentum · facetNormal >= 0)
// 이전 GPU 의 "flip microNormal" 은 sampling distribution bias.
// 반환: photon 향해 있는 microfacet normal (rotated to align with macro normal).
inline float3 SampleFacetNormal(thread RandomState& rng, float3 dir, float3 normal,
                                 float sigmaAlpha) {
    if (sigmaAlpha <= 0.0) return normal;
    // 로컬 frame: u, v 가 normal 에 수직.
    float3 up = abs(normal.y) < 0.999 ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 u = normalize(cross(up, normal));
    float3 v = cross(normal, u);
    const int maxOuter = 20;  // G4 무한, 우리 안전 cap
    for (int o = 0; o < maxOuter; o++) {
        float alpha = SampleMicrofacetAlpha(rng, sigmaAlpha);
        float sinA = MOP_SIN(alpha);
        float cosA = MOP_COS(alpha);
        float phi = TWO_PI * RandUniform(rng);
        float3 microNormal = normalize(cosA * normal + sinA * (MOP_COS(phi) * u + MOP_SIN(phi) * v));
        // G4: while (momentum * facetNormal >= 0) repeat
        //  ↔ accept = (dir · microNormal < 0) = (photon 이 facet 의 "front" 방향에서 옴)
        if (dot(dir, microNormal) < 0.0f) return microNormal;
    }
    return normal;  // fallback (rare)
}

// ============================================================
// UNIFIED 모델 표면 반사 방향 결정 — M5 수정
// ============================================================
inline float3 SampleUNIFIEDReflection(
    thread RandomState& rng,
    float3 dir, float3 normal,
    device const SurfaceGPU& surface,
    device const SurfaceLUT& surfLUT,  // Fix 71: SurfaceLUT
    float energy)
{
    // UNIFIED 모델: 4가지 반사 유형의 확률 분포
    // Fix 71: InterpolateProperty → InterpolateLUT (binary search 제거)
    float specSpikeProb = InterpolateLUT(surfLUT.specularSpike, energy);
    float specLobeProb = InterpolateLUT(surfLUT.specularLobe, energy);
    float backScatProb = InterpolateLUT(surfLUT.backScatter, energy);

    float lambertianProb = 1.0 - specSpikeProb - specLobeProb - backScatProb;
    if (lambertianProb < 0.0) lambertianProb = 0.0;

    float rand = RandUniform(rng);

    if (rand < specSpikeProb) {
        // Specular spike: 정반사
        float cosI = -dot(dir, normal);
        return normalize(dir + 2.0 * cosI * normal);
    }
    rand -= specSpikeProb;

    if (rand < specLobeProb) {
        // Specular lobe: 미세면 법선 기반 반사 — G4 DielectricMetal/DielectricDielectric
        // 의 do { GetFacetNormal; reflect; } while (newDir·normal <= 0) 정석 매칭.
        // 이전 "flip microNormal + maxBounce=5 fallback to spike" 는 sampling bias
        // 였음 (2026-05-16 발견: lobe σα=0.1 GPU/CPU sum 0.88 + peak 17cm shift).
        const int maxBounce = 20;  // G4 무한, 안전 cap
        for (int iter = 0; iter < maxBounce; iter++) {
            float3 microNormal = SampleFacetNormal(rng, dir, normal, surface.sigmaAlpha);
            float cosI = -dot(dir, microNormal);  // positive by SampleFacetNormal guarantee
            float3 newDir = normalize(dir + 2.0 * cosI * microNormal);
            // newDir 가 macro normal 의 upper hemisphere 에 있어야 유효 (G4 do-while
            // 조건과 동일). 아니면 microfacet 재샘플.
            if (dot(newDir, normal) > 0.0) return newDir;
        }
        // 포기 (rare): plain spike fallback
        float cosIplain = -dot(dir, normal);
        return normalize(dir + 2.0 * cosIplain * normal);
    }
    rand -= specLobeProb;

    if (rand < backScatProb) {
        // Backscatter: 입사 방향으로 반사
        return -dir;
    }

    // Lambertian: 코사인 가중 랜덤 반사
    return RandLambertian(rng, normal);
}

// 범용 closest-hit 갱신 판정 (모든 geometry 공통): coincident boundary 에서
// surface 있는 hit 을 surface 없는 hit 보다 우선 (surface 처리 보장 = G4 navigator 가
// surface boundary 를 통과 못 하게 하는 것과 동등). tolerance 는 scale-invariant
// (좌표 크기 비례 ULP + 최소값). 같은 위치(coincident)에 두 face 가 겹칠 때 surface
// 없는 face 가 fp32 연산차로 약간 먼저 hit 돼 surface(reflector 등)를 우회하던 버그 fix.
inline bool ShouldUpdateClosestHit(float tHit, uint surfId, float curDist, uint curSurfId) {
    float tol = fmax(fabs(tHit), fabs(curDist)) * 1e-5f + 1e-6f;
    bool coinc = (curDist < 1e19f) && (fabs(tHit - curDist) < tol);
    if (coinc) {
        bool newHasSurf = (surfId > 0u);
        bool curHasSurf = (curSurfId > 0u);
        if (newHasSurf && !curHasSurf) return true;   // surface 있는 hit 우선
        if (!newHasSurf && curHasSurf) return false;  // 기존 surface hit 유지
    }
    return (tHit < curDist);
}

// 점이 G4Polyhedra(N-sided, full phi, solid) 내부인지 판정 — mesh tri runtime
// sibling adjacency 에서 인접 polyhedra material 조회용. intersect 의 N lateral
// half-space(apothem = rMax·cos(π/N)) + axial cap 판정과 동일 식.
inline bool PointInPolyhedra(float3 pt, PolyhedraGeometry ph) {
    uint N = ph.numSides;
    if (N < 3) return false;
    float3 ctr = float3(ph.centerX, ph.centerY, ph.centerZ);
    float3 ax  = normalize(float3(ph.axisX, ph.axisY, ph.axisZ));
    float3 e1  = normalize(float3(ph.e1X, ph.e1Y, ph.e1Z));
    float3 e2  = normalize(float3(ph.e2X, ph.e2Y, ph.e2Z));
    float3 q = pt - ctr;
    if (abs(dot(q, ax)) > ph.halfLengthAxis) return false;
    float xl = dot(q, e1), yl = dot(q, e2);
    float angleStep = ph.phiTotal / (float)N;
    float apothem = ph.rMax * cos(M_PI_F / (float)N);
    for (uint k = 0; k < N; k++) {
        float ang = ph.phiStart + ((float)k + 0.5f) * angleStep;
        if (xl * cos(ang) + yl * sin(ang) > apothem) return false;
    }
    return true;
}

// ============================================================
// 표면 속성이 있는 경우의 경계 상호작용 처리 — M5 UNIFIED 모델 수정
// ============================================================
inline void ProcessBoundaryWithSurface(
    thread Photon& photon,
    thread RandomState& rng,
    float3 normal,
    device const SurfaceGPU& surface,
    device const SurfaceLUT& surfLUT,   // Fix 71: SurfaceLUT
    device const MaterialLUT* matLUTs,  // Fix 71: MaterialLUT for n1/n2
    uint matIdIn, uint matIdOut,
    device atomic_uint* diagCounters)   // 2026-05-06 진단: reflect dir stats
    // Fix 78d: matIn/matOut 파라미터 제거 — LUT로 굴절률 전환 이후 미사용
{
    #if ENABLE_DIAGNOSTICS
    atomic_fetch_add_explicit(&diagCounters[180], 1u, memory_order_relaxed);  // ProcessBoundaryWithSurface entries
    #endif
    // Fix 71: InterpolateProperty → InterpolateLUT for refractive indices
    float n1 = InterpolateLUT(matLUTs[matIdIn].rindex, photon.energy);
    float n2 = InterpolateLUT(matLUTs[matIdOut].rindex, photon.energy);
    if (n1 <= 0.0) n1 = 1.0;
    if (n2 <= 0.0) n2 = 1.0;

    float3 dir = float3(photon.direction);
    float3 pol = float3(photon.polarization);
    float cosI = -dot(dir, normal);
    if (cosI < 0.0) {
        normal = -normal;
        cosI = -cosI;
    }

    uint surfType = surface.type;
    uint surfFinish = surface.finish;
    uint surfModel = surface.model;

    // dielectric_metal: 반사 또는 흡수
    if (surfType == DIELECTRIC_METAL) {
        float reflProb = InterpolateLUT(surfLUT.reflectivity, photon.energy);

        float bndRand = RandUniform(rng);
        if (bndRand < reflProb) {
            // 반사 (dielectric_metal)
            photon.reflectedCount++;
            float3 newDir;
            if (surfModel == UNIFIED && surfFinish == GROUND) {
                newDir = SampleUNIFIEDReflection(rng, dir, normal, surface, surfLUT, photon.energy);
            } else {
                // S1 엄밀: dd_t reflect (torus mirror dominant path)
                newDir = reflect_direction(dir, normal);
            }
            // 2026-06-05 FIX: dielectric_metal 반사 편광 = G4 DoReflection() 정석
            //   `-pol + 2(pol·facetN)facetN` (specular). 기존 UpdatePolarizationReflect 는
            //   dielectric Fresnel amplitude → metal 엔 부적합. 단일반사(torus)선 무시됐으나
            //   B7 metal-wrap 다중반사(~20회)서 출사 각분포에 누적 → 재방출 잔여편차.
            //   facetN: polished=global normal(SpikeReflection), ground=micro-facet
            //   (G4 inline DoReflection: fFacetNormal=(newMom-oldMom).unit()).
            float3 facetN = (surfModel == UNIFIED && surfFinish == GROUND)
                              ? normalize(newDir - dir) : normal;
            float3 newPol = normalize(-pol + 2.0f * dot(pol, facetN) * facetN);
            photon.direction = packed_float3(newDir);
            photon.polarization = packed_float3(newPol);
            // Wächter-Binder offset_ray (2026-05-06 fix): metal reflection 후 ε push.
            // 이전 누락 → analytic SDF (torus) 의 self-intersection 으로 mirror 광자 lost.
            // Fresnel reflection 분기 와 동일한 push.
            // 정석 fix (2026-05-07): ε push 1e-4 → 1e-3. min-sd hybrid 의 kTangentTol(5e-4)
            // 보다 작으면 reflect 후 minSd 가 아직 tolerance 안 → 같은 surface 다시 false hit
            // → axis-aligned mirror 에서 무한 reflect (per-photon 9ms+).
            //
            // ★★ 중요 (2026-05-22, dielectric_metal reflector self-intersection): 이 ε push 는
            // CPU(TOPAS/G4 G4Box+skin) 대비 GPU 의 핵심 정확성 우위 지점이다. 이 push 를
            // 줄이거나 제거하면 GPU 도 CPU 처럼 self-intersection 으로 무너지니 절대 약화 금지.
            //   - 증상: CsI 결정을 reflector(dielectric_metal R=0.95)로 감싼 셋업에서 CPU 는
            //     reflect 직후 광자가 surface face 에 정확히 붙어 다음 step 이 ~2e-9 mm 로 줄고
            //     (StepTooSmall) G4OpBoundaryProcess 가 "OpBoun06: Boundary scattering may be
            //     incorrect" 를 내며 multi-bounce 를 못 한다 → CsI fluence 가 Reflectivity 와
            //     무관하게(R=0/0.5/0.95 모두 ~9e4) 죽어버림. (coincident/World size 무관 — G4Box
            //     자체의 reflect self-intersection 임을 plate gap·World 500mm 측정으로 확인.)
            //   - GPU: 이 push 로 self-intersection 회피 → reflect→multi-bounce 정상. R 따라
            //     monotonic (1.65e6/3.29e6/2.39e7), R=0.95 에서 ~14.5 bounce = 이론 12~20 범위.
            //   - 결과: reflector CsI fluence GPU/CPU = 263× (CPU under, GPU 정상). 이론 bounce 와
            //     일치하므로 CPU 비교 없이도 GPU 가 물리적으로 맞다. 상세: docs/topas_cpu_optical_root_cause_summary.md
            const float kEpsPushM = 1e-3f;  // 1 micron, > kTangentTol 5e-4
            // 2026-06-04: 평면 metal reflector self-intersection 회피 — 법선-only offset.
            //   multi-bounce tangential drift 제거(B7 z<-20 4873→45). torus 1.0000 검증.
            //   dielectric_dielectric(곡면 렌즈)는 forward clearance 필요해 newDir+normal 유지.
            photon.position = packed_float3(float3(photon.position) + (2.0f * kEpsPushM) * normal);
            #if ENABLE_DIAGNOSTICS
            // 2026-05-06 진단: dielectric_metal reflect dir/normal 평균 측정 (slot 140-149)
            // 누적 = (val+1)*1000, mean = sum/cnt/1000 - 1
            const float kReflStatSc = 1000.0f;
            uint nDx = (uint)((newDir.x + 1.0f) * kReflStatSc);
            uint nDy = (uint)((newDir.y + 1.0f) * kReflStatSc);
            uint nDz = (uint)((newDir.z + 1.0f) * kReflStatSc);
            uint nNx = (uint)((normal.x + 1.0f) * kReflStatSc);
            uint nNy = (uint)((normal.y + 1.0f) * kReflStatSc);
            uint nNz = (uint)((normal.z + 1.0f) * kReflStatSc);
            atomic_fetch_add_explicit(&diagCounters[140], nDx, memory_order_relaxed);
            atomic_fetch_add_explicit(&diagCounters[141], nDy, memory_order_relaxed);
            atomic_fetch_add_explicit(&diagCounters[142], nDz, memory_order_relaxed);
            atomic_fetch_add_explicit(&diagCounters[143], nNx, memory_order_relaxed);
            atomic_fetch_add_explicit(&diagCounters[144], nNy, memory_order_relaxed);
            atomic_fetch_add_explicit(&diagCounters[145], nNz, memory_order_relaxed);
            atomic_fetch_add_explicit(&diagCounters[146], 1u, memory_order_relaxed);
            // newDir.z 부호별 카운트
            if (newDir.z > 0.0f) atomic_fetch_add_explicit(&diagCounters[147], 1u, memory_order_relaxed);
            else atomic_fetch_add_explicit(&diagCounters[148], 1u, memory_order_relaxed);
            #endif
        } else {
            // 검출 또는 흡수
            if (surface.efficiency.count > 0) {
                float eff = InterpolateLUT(surfLUT.efficiency, photon.energy);
                if (RandUniform(rng) < eff) {
                    photon.status = DETECTED;
                    return;
                }
            }
            photon.status = BOUNDARY_ABS;
            return;
        }
        return;
    }

    // dielectric_dielectric UNIFIED ground:
    // G4 DielectricDielectric (G4OpBoundaryProcess.cc:1086-1093) —
    //   if (fFinish == polished) fFacetNormal = fGlobalNormal;
    //   else                     fFacetNormal = GetFacetNormal(...);
    // 즉 GetFacetNormal 의 do-while reject (facet 잘못 향하면 재샘플) 결과를
    // 그대로 facet normal 로 사용. Fresnel 은 이후 코드가 이 normal 로 수행.
    //
    // 2026-05-16 fix: dielectric_metal lobe path 와 같이 "flip + maxBounce=5
    // fallback" 패턴을 SampleFacetNormal (G4 do-while reject) 로 교체. 단일
    // 호출로 정석 facet 반환.
    if (surfModel == UNIFIED && surfFinish == GROUND && surface.sigmaAlpha > 0.0) {
        float3 microNormal = SampleFacetNormal(rng, dir, normal, surface.sigmaAlpha);
        normal = microNormal;
        cosI = -dot(dir, normal);
        // SampleFacetNormal 은 dot(dir, microNormal) < 0 보장 (G4 와 동일).
        // 단 fallback (rare) 시 macro normal 반환 — orientation 안전장치 유지.
        if (cosI < 0.0) { normal = -normal; cosI = -cosI; }
    }

    // 2026-05-08 BUG FIX: ProcessBoundaryFresnel 과 동일한 Wächter-Binder
    // offset_ray ε push 추가. dielectric_dielectric 분기 모두 (forceRefl,
    // reflProb-based, bare Fresnel) 에 self-intersection 차단을 위해 ε push
    // 적용. 누락 시 boundary 직후 같은 triangle 다시 hit → photon teleport
    // (STL biconvex lens 시 ~89% photon 손실, GPU/CPU=0.0316 root cause).
    const float kBoundaryPushEpsWS = 1e-4f;  // 100nm — ProcessBoundaryFresnel 과 동일 (L#1 1e-3, 1e-5 테스트 reject)

    // Fix 52: Geant4 UNIFIED 모델 dielectric_dielectric 로직 수정
    // Geant4 G4OpBoundaryProcess.cc line 463-488:
    //   rand > R + T          → DoAbsorption (표면 흡수)
    //   R < rand < R + T      → Transmission (방향 불변, 직선 투과)
    //   rand < R              → DielectricDielectric() (전체 Fresnel 계산)
    // 기존 GPU 코드는 rand < R일 때 직접 Lambertian 반사를 했으나,
    // Geant4는 Fresnel 계산을 수행 (결과는 반사 OR 투과).
    if (surface.reflectivity.count > 0) {
        #if ENABLE_DIAGNOSTICS
        atomic_fetch_add_explicit(&diagCounters[181], 1u, memory_order_relaxed);  // has reflectivity
        #endif
        float reflProb = InterpolateLUT(surfLUT.reflectivity, photon.energy);

        // TOPAS extension: ForceReflectivity strict mode.
        // b:Su/XXX/ForceReflectivity = "True" 시 reflProb 를 절대 반사 확률로
        // 사용 (rand<R: specular reflection, 아니면 Snell refraction). Geant4
        // default 의 Fresnel-gate 해석을 우회하여 의도된 R 을 정확히 강제하고
        // 싶을 때 (예: 가공된 거울 5 % 반사 정확히, 95 % 는 Snell 굴절로
        // 정상 진행).
        if (surface.forceReflectivity != 0u) {
            if (RandUniform(rng) < reflProb) {
                // Specular 반사 — UNIFIED ground 면 SampleUNIFIEDReflection
                photon.reflectedCount++;
                float3 newDir;
                if (surfModel == UNIFIED && surfFinish != POLISHED) {
                    newDir = SampleUNIFIEDReflection(rng, dir, normal, surface, surfLUT, photon.energy);
                } else {
                    newDir = reflect_direction(dir, normal);  // S1 엄밀
                }
                float3 newPol = UpdatePolarizationReflect(dir, newDir, pol, normal, n1, n2);
                photon.direction = packed_float3(newDir);
                photon.polarization = packed_float3(newPol);
                photon.position = packed_float3(float3(photon.position)
                    + (2.0f * kBoundaryPushEpsWS) * normal);
            } else {
                // Snell 굴절로 transmit (TIR 안전장치 포함)
                float ratio = n1 / n2;
                float sinT2 = ratio * ratio * (1.0 - cosI * cosI);
                if (sinT2 > 1.0) {
                    // 전반사 안전장치 — 강제 모드라도 물리적으로 굴절 불가
                    photon.reflectedCount++;
                    float3 newDir = reflect_direction(dir, normal);  // S1 엄밀
                    float3 newPol = UpdatePolarizationReflect(dir, newDir, pol, normal, n1, n2);
                    photon.direction = packed_float3(newDir);
                    photon.polarization = packed_float3(newPol);
                    photon.position = packed_float3(float3(photon.position)
                        + (2.0f * kBoundaryPushEpsWS) * normal);
                } else {
                    float cosT = MOP_SQRT(1.0 - sinT2);
                    float3 newDir = refract_direction(dir, ratio, cosI, cosT, normal);  // S1 엄밀
                    float3 newPol = UpdatePolarizationRefract(dir, newDir, pol, normal, n1, n2);
                    photon.direction = packed_float3(newDir);
                    photon.polarization = packed_float3(newPol);
                    photon.materialId = matIdOut;
                    photon.position = packed_float3(float3(photon.position)
                        - (2.0f * kBoundaryPushEpsWS) * normal);
                }
            }
            return;
        }

        float transProb = 0.0;
        if (surface.transmittance.count > 0)
            transProb = InterpolateLUT(surfLUT.transmittance, photon.energy);

        float rand = RandUniform(rng);

        if (rand > reflProb + transProb) {
            // 흡수/검출 (Geant4 DoAbsorption)
            if (surface.efficiency.count > 0 &&
                RandUniform(rng) < InterpolateLUT(surfLUT.efficiency, photon.energy)) {
                photon.status = DETECTED;
            } else {
                photon.status = BOUNDARY_ABS;
            }
            return;
        }

        if (rand > reflProb) {
            // Geant4 Transmission: 방향 불변, 물질만 전환 (Snell 굴절 아님!)
            photon.materialId = matIdOut;  // Fix 71: uint 직접 사용
            // 2026-05-08 fix: 방향 불변 transmission 도 boundary 통과 후 self-intersect
            // 차단 push (-normal: 새 매질 쪽). dir 으로 그대로 진행하므로 -normal 으로 push.
            photon.position = packed_float3(float3(photon.position)
                - (2.0f * kBoundaryPushEpsWS) * normal);
            return;
        }

        // rand < reflProb: Geant4 DielectricDielectric() 호환
        // 전체 Fresnel 계산 수행 (결과: 반사 또는 굴절)
        {
            float Rs, Rp;
            float R = FresnelReflectance(cosI, n1, n2, dir, normal, pol, Rs, Rp);

            if (RandUniform(rng) < R) {
                // Fresnel 반사 → UNIFIED 모델 반사 방향 적용
                photon.reflectedCount++;
                float3 newDir;
                if (surfModel == UNIFIED && surfFinish != POLISHED) {
                    newDir = SampleUNIFIEDReflection(rng, dir, normal, surface, surfLUT, photon.energy);
                } else {
                    newDir = reflect_direction(dir, normal);  // S1 엄밀
                }
                float3 newPol = UpdatePolarizationReflect(dir, newDir, pol, normal, n1, n2);
                photon.direction = packed_float3(newDir);
                photon.polarization = packed_float3(newPol);
                // Fix 62b: TIR(R=1.0) 후 외부 재질 전환 (WithSurface Fresnel 경로)
                #if EMULATE_G4_TIR_PUSH
                if (R >= 1.0) {
                    photon.materialId = matIdOut;  // Fix 71: uint 직접 사용
                }
                #endif
                photon.position = packed_float3(float3(photon.position)
                    + (2.0f * kBoundaryPushEpsWS) * normal);
            } else {
                // Fresnel 굴절 (Snell의 법칙)
                float ratio = n1 / n2;
                float sinT2 = ratio * ratio * (1.0 - cosI * cosI);

                if (sinT2 > 1.0) {
                    // 전반사 (TIR) → UNIFIED 반사 방향
                    photon.reflectedCount++;
                    float3 newDir;
                    if (surfModel == UNIFIED && surfFinish != POLISHED) {
                        newDir = SampleUNIFIEDReflection(rng, dir, normal, surface, surfLUT, photon.energy);
                    } else {
                        newDir = reflect_direction(dir, normal);  // S1 엄밀
                    }
                    float3 newPol = UpdatePolarizationReflect(dir, newDir, pol, normal, n1, n2);
                    photon.direction = packed_float3(newDir);
                    photon.polarization = packed_float3(newPol);
                    // Fix 62b: TIR 안전장치 분기 외부 재질 전환
                    #if EMULATE_G4_TIR_PUSH
                    photon.materialId = matIdOut;  // Fix 71: uint 직접 사용
                    #endif
                    photon.position = packed_float3(float3(photon.position)
                        + (2.0f * kBoundaryPushEpsWS) * normal);
                } else {
                    float cosT = MOP_SQRT(1.0 - sinT2);
                    float3 newDir = refract_direction(dir, ratio, cosI, cosT, normal);  // S1 엄밀
                    float3 newPol = UpdatePolarizationRefract(dir, newDir, pol, normal, n1, n2);
                    photon.direction = packed_float3(newDir);
                    photon.polarization = packed_float3(newPol);
                    photon.materialId = matIdOut;  // Fix 71: uint 직접 사용
                    photon.position = packed_float3(float3(photon.position)
                        - (2.0f * kBoundaryPushEpsWS) * normal);
                }
            }
            return;
        }
    }

    // 표면 정의 반사율이 없으면 Fresnel (편광 의존)
    float Rs, Rp;
    float R = FresnelReflectance(cosI, n1, n2, dir, normal, pol, Rs, Rp);

    if (RandUniform(rng) < R) {
        photon.reflectedCount++;
        #if ENABLE_DIAGNOSTICS
        atomic_fetch_add_explicit(&diagCounters[173], 1u, memory_order_relaxed);  // lens WS Fresnel reflect
        #endif
        float3 newDir = reflect_direction(dir, normal);  // S1 엄밀
        float3 newPol = UpdatePolarizationReflect(dir, newDir, pol, normal, n1, n2);
        photon.direction = packed_float3(newDir);
        photon.polarization = packed_float3(newPol);
        // Fix 62b: WithSurface no-reflectivity 경로 TIR push
        #if EMULATE_G4_TIR_PUSH
        if (R >= 1.0) {
            photon.materialId = matIdOut;  // Fix 71: uint 직접 사용
        }
        #endif
        photon.position = packed_float3(float3(photon.position)
            + (2.0f * kBoundaryPushEpsWS) * normal);
    } else {
        float ratio = n1 / n2;
        float sinT2 = ratio * ratio * (1.0 - cosI * cosI);

        if (sinT2 > 1.0) {
            photon.reflectedCount++;
            #if ENABLE_DIAGNOSTICS
            atomic_fetch_add_explicit(&diagCounters[174], 1u, memory_order_relaxed);  // lens WS TIR
            #endif
            float3 newDir = reflect_direction(dir, normal);  // S1 엄밀
            float3 newPol = UpdatePolarizationReflect(dir, newDir, pol, normal, n1, n2);
            photon.direction = packed_float3(newDir);
            photon.polarization = packed_float3(newPol);
            #if EMULATE_G4_TIR_PUSH
            photon.materialId = matIdOut;  // Fix 71: uint 직접 사용
            #endif
            photon.position = packed_float3(float3(photon.position)
                + (2.0f * kBoundaryPushEpsWS) * normal);
        } else {
            #if ENABLE_DIAGNOSTICS
            atomic_fetch_add_explicit(&diagCounters[175], 1u, memory_order_relaxed);  // lens WS transmit
            #endif
            float cosT = MOP_SQRT(1.0 - sinT2);
            float3 newDir = refract_direction(dir, ratio, cosI, cosT, normal);  // S1 엄밀
            float3 newPol = UpdatePolarizationRefract(dir, newDir, pol, normal, n1, n2);
            photon.direction = packed_float3(newDir);
            photon.polarization = packed_float3(newPol);
            photon.materialId = matIdOut;  // Fix 71: uint 직접 사용
            photon.position = packed_float3(float3(photon.position)
                - (2.0f * kBoundaryPushEpsWS) * normal);
            // Fix 56: Surface Fresnel refraction nudge 제거 (tmin=1e-4가 경계 탈출 보장)
            // photon.position = packed_float3(float3(photon.position) + 2e-4 * newDir);
        }
    }
}

// ============================================================
// 메인 전파 커널
// ============================================================
kernel void propagatePhotons(
    device Photon*              photons     [[buffer(0)]],
    device const MaterialGPU*   materials   [[buffer(1)]],
    device const SurfaceGPU*    surfaces    [[buffer(2)]],
    constant SimConfig&         config      [[buffer(3)]],
    device Hit*                 hits        [[buffer(4)]],
    device atomic_uint&         hitCount    [[buffer(5)]],
    device const TriangleAttr*  triangles   [[buffer(6)]],  // Fix 71: 32B 경량 구조체
    device const uint&          numTriangles [[buffer(7)]],
    device const uint&          totalPhotons [[buffer(8)]],
    primitive_acceleration_structure accelStruct [[buffer(9)]],
    device TransitHit*          transitHits [[buffer(10)]],  // Fix 41: 통과 기록 버퍼
    device atomic_uint&         transitCount [[buffer(11)]],  // Fix 41: 통과 카운터
    device const uint&          maxTransitHits [[buffer(12)]],// Fix 41: 버퍼 크기
    // 2026-05-07: 항상 선언 (ENABLE_DIAGNOSTICS=0 시 함수 시그니처 호환).
    // 사용은 #if ENABLE_DIAGNOSTICS 로 가드. host 가 buffer 항상 binding.
    device atomic_uint*         diagCounters  [[buffer(13)]],
    device TrajectoryStep*      trajectoryBuf [[buffer(14)]],
    device atomic_uint&         trajectoryCount [[buffer(15)]],
    device const MaterialLUT*   matLUTs [[buffer(16)]],       // Fix 64: 재질 LUT
    device const SurfaceLUT*    surfLUTs [[buffer(17)]],      // Fix 71: 표면 LUT
    device const PhotonMeta*    photonMeta [[buffer(18)]],    // Fix 78측정-6: cold meta (hit 기록 시만)
    device const BoxGeometry*   boxGeometries [[buffer(21)]], // G1: analytic G4Box
    device const uint&          numBoxGeometries [[buffer(22)]],
    // SF1 (2026-04-24): GPU surface flux scorer buffers
    device const SurfaceDef*    surfaceDefs [[buffer(23)]],   // SF1: registered surface planes
    device SurfaceHit*          surfaceHits [[buffer(24)]],   // SF1: crossing event buffer
    device atomic_uint&         surfaceHitCount [[buffer(25)]], // SF1: crossing counter
    device const uint&          maxSurfaceHits [[buffer(26)]], // SF1: buffer capacity
    // Phase 1 (2026-05-05): G4Sphere analytic intersection (in-kernel loop)
    device const SphereGeometry* sphereGeometries [[buffer(27)]],
    // Phase 4 (2026-05-18): slot 28 = PolyhedraGeometry buffer (sphere count 는 cylTorPolyCounts.w 에 packing)
    device const PolyhedraGeometry* polyhedraGeometries [[buffer(28)]],
    // Phase 2 (2026-05-05): G4Tubs analytic intersection (in-kernel loop)
    device const CylinderGeometry* cylinderGeometries [[buffer(29)]],
    // 통합 count buffer (uint4): {numCyl, numTor, numPoly, numSphere}
    device const uint4& cylTorPolyCounts [[buffer(30)]],
    // Phase 3 (2026-05-05): G4Torus analytic intersection (Ferrari quartic)
    // buffer 19 사용 — 31, 32 는 Metal 한도 초과
    device const TorusGeometry* torusGeometries [[buffer(19)]],
    uint gid [[thread_position_in_grid]])
{
    // dispatchThreads로 정확한 스레드 수 디스패치
    uint idx = gid;
    if (idx >= totalPhotons) return;


    // 히트 버퍼 최대 크기
    uint maxHits = totalPhotons;

    // Fix 41b: 스코어링 AABB
    bool scoringAABBEnabled = (config.scoringAABBEnabled != 0);
    float3 scoringMin = float3(config.scoringAABBMinX, config.scoringAABBMinY, config.scoringAABBMinZ);
    float3 scoringMax = float3(config.scoringAABBMaxX, config.scoringAABBMaxY, config.scoringAABBMaxZ);

    Photon photon = photons[idx];
    if (photon.status != ALIVE) {
        photons[idx] = photon;
        return;
    }

    // RanecuEngine 시드 초기화 (kernelSalt=4: propagation)
    // 2026-05-12 fix: idx + globalPhotonOffset 로 batch 마다 unique RNG (RNG repetition bug fix)
    RandomState rng = InitRandomState(idx + config.globalPhotonOffset, config.randomSeed, 4u);

    uint currentMatId = photon.materialId;
    #if ENABLE_DIAGNOSTICS
    int lastTriIdx = -1;  // 진단: 이전 step에서 충돌한 삼각형 ID (ENABLE_DIAGNOSTICS 전용)
    #endif

    // Fix 64c: intersector를 루프 밖에서 한 번만 생성
    // Fix 78a: HW RT 힌트 3종 — triangle-only geometry, opaque, triangle_data 제거
    intersector<> hwrt_inter;
    hwrt_inter.accept_any_intersection(false);
    hwrt_inter.assume_geometry_type(geometry_type::triangle);
    hwrt_inter.force_opacity(forced_opacity::opaque);
    #if ENABLE_DIAGNOSTICS
    // First-reflect 광자 추적용 — 처음 N개 광자의 모든 step 을 dump,
    // CPU 측에서 photonIdx 기준 그룹화 후 first reflect 광자 식별.
    bool dumpTrajectory = (idx < TRAJ_DUMP_PHOTON_LIMIT);
    #endif

    // 2026-05-26 최적화: absLen spline 캐싱. absLen=f(energy,material)은 한 물질
    // 구간 안에서 불변 (TIR 바운스도 같은 물질 → 매 step 재평가는 redundant).
    // material 또는 energy 변경 시에만 SplineEvalG4Packed 호출 → G4 spline 비트 동일,
    // 호출 횟수만 N→(물질 segment 수)로 감소. RandUniform 은 매 step 그대로 (RNG 불변).
    float cachedAbsLen    = -1.0f;
    uint  cachedAbsLenMat = 0xFFFFFFFFu;
    float cachedAbsLenE   = -1.0f;

    for (uint step = 0; step < config.maxStepsPerPhoton; step++) {
        if (photon.status != ALIVE) break;

        // SF1 (post-boundary refactor 2026-04-24): step 시작 위치/material 캡처
        // step 끝에서 boundary 처리 후 비교 → CPU TsScoreSurfaceTrackCount 의미론
        float3 sf1_stepStartPos = float3(photon.position);
        float3 sf1_stepStartDir = float3(photon.direction);  // pre-boundary direction
        (void)sf1_stepStartDir;  // 2026-05-12: -Werror unused 회피 (post-boundary refactor 후 미사용)
        uint   sf1_stepStartMat = currentMatId;

        // [디버그 코드 제거됨 — Fix 37/38 검증 완료]

        // Fix: materialId 범위 검증 — 잘못된 ID로 버퍼 오버리드 방지
        if (currentMatId >= MAX_MATERIALS) {
            photon.status = ABSORBED;
            break;
        }

        device const MaterialGPU& mat = materials[currentMatId];

        // =======================================================
        // C5 수정: 프로세스 경쟁 모델 — 모든 프로세스가 독립적으로 상호작용 길이 샘플링
        // 최단 거리 프로세스가 승리
        // =======================================================

        // 각 프로세스의 상호작용 길이 샘플링
        float distToAbsorb = 1e20;
        float distToRayleigh = 1e20;
        float distToMie = 1e20;
        float distToWLS = 1e20;

        // 흡수 (Fix 78b: function_constant → dead branch 제거)
        if (kEnableAbsorption) {
            // 2026-05-24 정확 G4 spline eval (packed knots, LUT 보간오차 0) 유지.
            // 2026-05-26: material/energy 불변이면 캐시 사용 → 값 비트 동일, 호출만 절감.
            if (currentMatId != cachedAbsLenMat || photon.energy != cachedAbsLenE) {
                cachedAbsLen = SplineEvalG4Packed(matLUTs[currentMatId].absLen, photon.energy);
                if (cachedAbsLen <= 0.0f) cachedAbsLen = 1e-3f;  // 음수 overshoot → 즉시흡수
                cachedAbsLenMat = currentMatId;
                cachedAbsLenE   = photon.energy;
            }
            float absRand = RandUniform(rng);
            distToAbsorb = -cachedAbsLen * MOP_LOG(max(absRand, 1e-10f));
        }

        // 레일리 산란
        if (kEnableRayleigh) {
            float rayLen = InterpolateLUT(matLUTs[currentMatId].rayLen, photon.energy);
            if (rayLen > 0.0)
                distToRayleigh = -rayLen * MOP_LOG(max(RandUniform(rng), 1e-10f));
        }

        // 미 산란
        if (kEnableMie) {
            float mieLen = InterpolateLUT(matLUTs[currentMatId].mieLen, photon.energy);
            if (mieLen > 0.0)
                distToMie = -mieLen * MOP_LOG(max(RandUniform(rng), 1e-10f));
        }

        // WLS 흡수
        if (kEnableWLS) {
            // 2026-06-04 fix: InterpolateLUT(256점 LUT) → SplineEvalG4Packed (raw 점 per-photon
            //   직접 eval, LUT 이산화오차 0). WLSABSLENGTH 의 가파른 절벽(예: 1e8→2.0mm @2.25eV)을
            //   256점 LUT(빈 0.055eV)가 뭉개 CsI 발광 2.25eV 서 투명값 → GPU under-absorb.
            //   CPU(G4 raw linear, N<5 spline 비활성)와 동일하게 raw 점 직접 평가로 절벽 보존.
            //   absLen 과 동일 방식(fillAbsLenPacked). N<5 시 D2=0 → linear-on-raw (CPU 일치).
            float wlsLen = SplineEvalG4Packed(matLUTs[currentMatId].wlsLen, photon.energy);
            if (wlsLen > 0.0)
                distToWLS = -wlsLen * MOP_LOG(max(RandUniform(rng), 1e-10f));
        }

        // 지오메트리 교차 검사 (Metal Hardware Ray Tracing BVH)
        float distToBoundary = 1e20;
        uint closestSurfId = 0u;  // 범용 coincident: 현재 closest hit 의 surfaceId (surface 있는 hit 우선)
        int closestTriIdx = -1;
        float3 closestNormal = float3(0);
        {
            // BVH grazing hit reject + next-hit 재호출 (Apple Metal HW RT).
            // Geant4 G4Navigator는 |cosI|<eps + face 위 광자를 그 면에서 무시하고
            // 다음 face가 distance 결정. GPU에서 동등 처리: ray.tmin update +
            // 재호출로 다음 hit 찾기. max 5 retry로 무한루프 방지.
            // BVH 는 centroid-shifted vertices 사용 — ray origin 도 동일 shift 로
            // BVH-local 좌표계에서 계산 → fp32 ULP 가 |coord| 비례 정밀도 개선.
            float3 bvhCentroid = float3(config.bvhCentroidX, config.bvhCentroidY, config.bvhCentroidZ);
            ray r(float3(photon.position) - bvhCentroid, float3(photon.direction), 1e-7, 1e10);
            const int maxBVHRetries = 5;
            for (int retry = 0; retry < maxBVHRetries; retry++) {
                auto result = hwrt_inter.intersect(r, accelStruct);
                if (result.type != intersection_type::triangle) break;
                // Chroma 정석 fix (L1 expert): 직전 hit triangle reject 로
                // self-intersection 누설 차단 (multi-bounce reflect 47% over fix).
                if ((int)result.primitive_id == photon.lastHitTriId) {
                    r.min_distance = result.distance + 1e-4f;
                    if (r.min_distance >= r.max_distance) break;
                    continue;
                }
                float3 testNormal = float3(triangles[result.primitive_id].normal);
                float3 snapped = testNormal;
                const float kSnapEps = 1e-5f;
                if (abs(testNormal.x) > 1.0f - kSnapEps && abs(testNormal.y) < kSnapEps && abs(testNormal.z) < kSnapEps)
                    snapped = float3(testNormal.x > 0 ? 1.0f : -1.0f, 0, 0);
                else if (abs(testNormal.y) > 1.0f - kSnapEps && abs(testNormal.x) < kSnapEps && abs(testNormal.z) < kSnapEps)
                    snapped = float3(0, testNormal.y > 0 ? 1.0f : -1.0f, 0);
                else if (abs(testNormal.z) > 1.0f - kSnapEps && abs(testNormal.x) < kSnapEps && abs(testNormal.y) < kSnapEps)
                    snapped = float3(0, 0, testNormal.z > 0 ? 1.0f : -1.0f);
                float cosCheck = abs(dot(float3(photon.direction), snapped));
                if (cosCheck <= 1e-4f) {
                    r.min_distance = result.distance + 1e-4f;
                    if (r.min_distance >= r.max_distance) break;
                    continue;
                }
                photon.lastHitTriId = (int)result.primitive_id;
                distToBoundary = result.distance;
                closestTriIdx = (int)result.primitive_id;
                closestSurfId = triangles[result.primitive_id].surfaceId;
                closestNormal = snapped;
                break;
            }
        }

        // ========================================================
        // G1 (정석 fix, Opticks reference): G4Box analytic slab intersect.
        // BVH 메시 결과보다 작은 t 값 찾으면 distToBoundary/closestNormal 교체.
        // ========================================================
        bool   boxHitFlag = false;
        int    boxHitIdx = -1;
        for (uint b = 0; b < numBoxGeometries; b++) {
            BoxGeometry box = boxGeometries[b];
            float3 ctr = float3(box.center);
            float3 hl = float3(box.HLX, box.HLY, box.HLZ);
            float3 pl = float3(photon.position) - ctr;
            float3 d  = float3(photon.direction);
            // O4 Patch B v2: inside 판정에 surface tolerance를 +방향으로 적용.
            // 표면 위에 있거나 막 통과한 광자도 inside로 간주 → tFar (DistanceToOut) 사용
            // = self-intersection 회피. Geant4 G4Box::Inside는 surface일 때 kSurface 반환,
            // 이는 G4Navigator가 DistanceToOut 호출하도록 함.
            const float kSurfTol = 1e-4f;
            bool insideBox = (abs(pl.x) <= hl.x + kSurfTol &&
                              abs(pl.y) <= hl.y + kSurfTol &&
                              abs(pl.z) <= hl.z + kSurfTol);
            float tNear = -1e20, tFar = 1e20;
            int   nearAxis = -1, farAxis = -1;
            float nearSign = 0.0, farSign = 0.0;
            // x
            if (abs(d.x) < 1e-20f) {
                if (pl.x < -hl.x || pl.x > hl.x) { tNear = 1e20; tFar = -1e20; }
            } else {
                float invd = 1.0f / d.x;
                float t1 = (-hl.x - pl.x) * invd;
                float t2 = ( hl.x - pl.x) * invd;
                float tmin = min(t1, t2); float tmax = max(t1, t2);
                float signMin = (t1 < t2) ? -1.0f : 1.0f;
                float signMax = (t1 < t2) ?  1.0f : -1.0f;
                if (tmin > tNear) { tNear = tmin; nearAxis = 0; nearSign = signMin; }
                if (tmax < tFar)  { tFar  = tmax; farAxis  = 0; farSign  = signMax; }
            }
            // y
            if (tNear <= tFar) {
                if (abs(d.y) < 1e-20f) {
                    if (pl.y < -hl.y || pl.y > hl.y) { tNear = 1e20; tFar = -1e20; }
                } else {
                    float invd = 1.0f / d.y;
                    float t1 = (-hl.y - pl.y) * invd;
                    float t2 = ( hl.y - pl.y) * invd;
                    float tmin = min(t1, t2); float tmax = max(t1, t2);
                    float signMin = (t1 < t2) ? -1.0f : 1.0f;
                    float signMax = (t1 < t2) ?  1.0f : -1.0f;
                    if (tmin > tNear) { tNear = tmin; nearAxis = 1; nearSign = signMin; }
                    if (tmax < tFar)  { tFar  = tmax; farAxis  = 1; farSign  = signMax; }
                }
            }
            // z
            if (tNear <= tFar) {
                if (abs(d.z) < 1e-20f) {
                    if (pl.z < -hl.z || pl.z > hl.z) { tNear = 1e20; tFar = -1e20; }
                } else {
                    float invd = 1.0f / d.z;
                    float t1 = (-hl.z - pl.z) * invd;
                    float t2 = ( hl.z - pl.z) * invd;
                    float tmin = min(t1, t2); float tmax = max(t1, t2);
                    float signMin = (t1 < t2) ? -1.0f : 1.0f;
                    float signMax = (t1 < t2) ?  1.0f : -1.0f;
                    if (tmin > tNear) { tNear = tmin; nearAxis = 2; nearSign = signMin; }
                    if (tmax < tFar)  { tFar  = tmax; farAxis  = 2; farSign  = signMax; }
                }
            }
            if (!(tNear <= tFar)) continue;
            // O4 Patch B: inside/outside 분기 (Geant4 G4Box DistanceToOut/In 동등)
            // - inside면 tFar (DistanceToOut), 광자가 box exit
            // - outside면 tNear (DistanceToIn), 광자가 box entry
            // - outside + tNear<eps인 경우 광자가 표면 위에 있음 → reject
            //   (BVH triangle hit 또는 다른 box에 의존). tFar fallback은 box 반대편 hit
            //   생성하는 버그 유발하므로 사용 금지.
            float tHit; int hitAxis; float hitSign;
            if (insideBox) {
                if (tFar < kSurfTol) continue;
                tHit = tFar; hitAxis = farAxis; hitSign = farSign;
            } else {
                if (tNear < kSurfTol) continue;
                tHit = tNear; hitAxis = nearAxis; hitSign = nearSign;
            }
            if (ShouldUpdateClosestHit(tHit, box.surfaceId, distToBoundary, closestSurfId)) {
                // 2026-05-14 정석 nested volume 처리 (G4 navigator daughter 우선 동등):
                // box analytic 은 child volume (Hole 등) 을 subtraction 하지 않음.
                // box hit point 가 child cylinder (cyl.materialIdOutside == box.matIn,
                // 즉 cyl 가 이 box 안에 위치) 안이면 그 hit 는 child 영역 — box surface
                // 처리하면 안 됨 (G4 는 daughter volume boundary 가 진짜).
                // → box hit invalid, cylinder analytic 이 child boundary 처리.
                float3 boxHitPt = float3(photon.position) + tHit * float3(photon.direction);
                bool inChildVolume = false;
                // child cylinder 가 box 를 z-관통하면 (Hole HL == box HLZ) box 의
                // +z/-z 면 hit point 가 cylinder 의 +z/-z plane 과 coincident 다.
                // strict `<` 는 그 경계를 child 영역으로 인식 못 함 → box hit 이
                // child(Hole) 영역에서 잘못 등록됨. tolerance 로 coincident 경계 포함.
                const float kChildTol = 1e-3f;
                for (uint cc = 0; cc < cylTorPolyCounts.x; cc++) {
                    CylinderGeometry childCyl = cylinderGeometries[cc];
                    if (childCyl.materialIdOutside != box.materialIdInside) continue;  // not a child of this box
                    float3 cq = boxHitPt - float3(childCyl.center);
                    float cr2 = cq.x * cq.x + cq.y * cq.y;
                    float cR  = childCyl.radius;
                    if (cr2 < cR * cR + 2.0f * cR * kChildTol &&
                        abs(cq.z) < childCyl.halfLength + kChildTol) {
                        inChildVolume = true;
                        break;
                    }
                }
                if (!inChildVolume) {
                    distToBoundary = tHit;
                    closestTriIdx = -1;
                    float3 n = float3(0);
                    if (hitAxis == 0) n.x = hitSign;
                    else if (hitAxis == 1) n.y = hitSign;
                    else                   n.z = hitSign;
                    closestNormal = n;
                    boxHitFlag = true;
                    boxHitIdx = (int)b;
                    closestSurfId = box.surfaceId;
                }
            }
        }

        // ========================================================
        // Phase 1 (2026-05-05, Opticks 정석): G4Sphere analytic intersection.
        // CPU G4Sphere::DistanceToOut/In (quadratic) 동등. mesh 우회.
        // Ray: P + t·d, Sphere: |P + t·d - C|² = R²  → t² + 2(d·q)t + (q²-R²) = 0
        //   q = P - C, h = d·q (d unit), disc = h² - (q² - R²)
        //   t1 = -h - √disc (entry), t2 = -h + √disc (exit)
        // ========================================================
        bool sphereHitFlag = false;
        int  sphereHitIdx  = -1;
        for (uint s = 0; s < cylTorPolyCounts.w; s++) {
            SphereGeometry sph = sphereGeometries[s];
            float3 ctr = float3(sph.center);
            float  R   = sph.radius;
            float  Rmin = sph.innerRadius;  // Phase 1.1 (2026-05-16): 0 = full, >0 = shell
            float3 q   = float3(photon.position) - ctr;
            float3 d   = float3(photon.direction);
            float  qq  = dot(q, q);
            float  R2  = R * R;
            float  h   = dot(d, q);
            float  disc_o = h * h - (qq - R2);
            const float kSurfTol = 1e-4f;
            bool hasInner = (Rmin > kSurfTol);

            // outer sphere intersection
            if (disc_o < 0.0f) continue;  // ray misses outer sphere → no hit
            float sqD_o = sqrt(disc_o);
            float to_in  = -h - sqD_o;
            float to_out = -h + sqD_o;

            // inner sphere intersection (RMin>0 시)
            float ti_in = 1e20f, ti_out = -1e20f;
            bool innerHits = false;
            if (hasInner) {
                float Rmin2  = Rmin * Rmin;
                float disc_i = h * h - (qq - Rmin2);
                if (disc_i >= 0.0f) {
                    float sqD_i = sqrt(disc_i);
                    ti_in  = -h - sqD_i;
                    ti_out = -h + sqD_i;
                    innerHits = true;
                }
            }

            // 광자 위치 분류 (kSurfTol 안쪽까지 shell 로 포함).
            bool inOuter = (qq < R2 + 2.0f * R * kSurfTol);
            bool inInner = hasInner ? (qq < Rmin*Rmin - 2.0f * Rmin * kSurfTol) : false;
            // Phase 1.2 fix: hemisphere case 의 inside test 에 theta range 도 적용.
            // 광자 sphere radial 안 (qq<R²) 이라도 hemisphere 의 theta 밖 (예 lower
            // hemisphere 시 광자 z<0) 이면 inShell 분류하면 안 됨 — 광자 actually
            // hemisphere volume 밖.
            // q.z 가 hemisphere 안인지 검사 (RMin=0 인 경우는 sphere center 포함 OK).
            float photonR = (qq > 0.0f) ? sqrt(qq) : 1e-10f;
            float photonCosTh = clamp(q.z / photonR, -1.0f, 1.0f);
            float photonTh = acos(photonCosTh);
            bool inThetaRange = true;
            if (sph.deltaTheta < 3.14f) {
                float thS = sph.thetaStart;
                float thE = sph.thetaStart + sph.deltaTheta;
                inThetaRange = (photonTh >= thS - kSurfTol)
                            && (photonTh <= thE + kSurfTol);
            }
            bool inShell = inOuter && !inInner && inThetaRange;
            bool inHole  = inInner && inThetaRange;

            // Phase 1.2 (2026-05-16): theta range support (hemisphere etc).
            // 광자 hit 점의 theta = acos((hit.z - center.z) / r) 가
            // [thetaStart, thetaStart + deltaTheta] 안인지 검사. 밖 → skip.
            // full sphere (deltaTheta ≈ π) 인 경우 자동 통과.
            float thStart = sph.thetaStart;
            float thEnd   = sph.thetaStart + sph.deltaTheta;
            bool isFullTheta = (sph.deltaTheta >= 3.14f);  // π 거의 같음

            float tHit = 1e20f;
            int   hitWhich = -1;
            // candidate t list (helper)
            float cands[4] = { to_in, to_out, ti_in, ti_out };
            int   whichs[4] = { 0, 0, 1, 1 };
            int   nCand = innerHits ? 4 : 2;
            for (int i = 0; i < nCand; i++) {
                float t = cands[i];
                if (t < kSurfTol) continue;
                if (t >= tHit)   continue;
                // 광자 위치 분류 기반 valid 검사
                int w = whichs[i];
                bool valid = false;
                if (inShell) {
                    // shell 안: exit candidate (to_out, ti_in 이 nearest exit)
                    if (w == 0 && abs(t - to_out) < kSurfTol) valid = true;
                    if (w == 1 && abs(t - ti_in)  < kSurfTol) valid = true;
                } else if (inHole) {
                    if (w == 1 && abs(t - ti_out) < kSurfTol) valid = true;
                } else { // outside
                    if (w == 0 && abs(t - to_in)  < kSurfTol) valid = true;
                }
                if (!valid) continue;
                // theta range 검증 (hemisphere 등)
                if (!isFullTheta) {
                    float3 hp = q + t * d;
                    float rHit = (w == 0) ? R : Rmin;
                    float cosTh = clamp(hp.z / rHit, -1.0f, 1.0f);
                    float thHit = acos(cosTh);
                    if (thHit < thStart - kSurfTol || thHit > thEnd + kSurfTol) {
                        continue;  // hit 점이 theta range 밖 → invalid
                    }
                }
                tHit = t;
                hitWhich = w;
            }
            if (hitWhich < 0) continue;

            if (ShouldUpdateClosestHit(tHit, sph.surfaceId, distToBoundary, closestSurfId)) {
                distToBoundary = tHit;
                closestTriIdx = -1;
                // Outward normal:
                //   outer hit: +radial (sphere center → hit)
                //   inner hit: -radial (away from shell material, into hole)
                float3 hitPos = float3(photon.position) + tHit * d;
                if (hitWhich == 0) {
                    closestNormal = (hitPos - ctr) / R;
                } else {
                    closestNormal = -(hitPos - ctr) / Rmin;
                }
                boxHitFlag = false;
                sphereHitFlag = true;
                sphereHitIdx = (int)s;
                closestSurfId = sph.surfaceId;
            }
        }

        // ========================================================
        // Phase 2 (2026-05-05): G4Tubs (TsCylinder) analytic intersection.
        // Z-axis aligned: outer radius RMax, optional inner radius RMin (tube/shell).
        // Phase 2.1 (2026-05-16): RMin>0 (tube/shell) 지원.
        //   - Outer side ray-quadratic in xy: r² = RMax²
        //   - Inner side ray-quadratic in xy: r² = RMin²  (RMin>0 시)
        //   - Z slab |z| ≤ HL
        // 광자 위치 3 영역 분류 후 최초 boundary hit 결정 (G4Tubs::DistanceToOut/In 등가):
        //   in-shell (Rmin ≤ r ≤ Rmax, |z| ≤ HL): nearest exit (outer / inner / z-plane)
        //   in-hole  (r < Rmin):                    nearest inner-cyl exit
        //   outside  (r > Rmax 또는 |z| > HL):       nearest outer entry
        // ========================================================
        bool cylHitFlag = false;
        int  cylHitIdx  = -1;
        bool cylHitIsZPlane = false;   // hit normal 이 z-plane 인지 (true) 또는 측면 (false)
        for (uint c = 0; c < cylTorPolyCounts.x; c++) {
            CylinderGeometry cyl = cylinderGeometries[c];
            float3 ctr  = float3(cyl.center);
            float  R    = cyl.radius;
            float  Rmin = cyl.innerRadius;       // 0 = full cylinder, >0 = tube/shell
            float  HL   = cyl.halfLength;
            float3 q    = float3(photon.position) - ctr;
            float3 d    = float3(photon.direction);
            const float kSurfTol = 1e-4f;
            bool   hasInner = (Rmin > kSurfTol);

            // --- Z slab (top/bottom planes) ---
            float tz_in = -1e20f, tz_out = 1e20f;
            if (abs(d.z) < 1e-20f) {
                if (q.z < -HL - kSurfTol || q.z > HL + kSurfTol) continue;
            } else {
                float invd = 1.0f / d.z;
                float t1 = (-HL - q.z) * invd;
                float t2 = ( HL - q.z) * invd;
                tz_in  = min(t1, t2);
                tz_out = max(t1, t2);
            }

            // --- Outer side quadratic: a*t² + b*t + c_o = 0 (c_o = q.r² - Rmax²) ---
            // 2026-05-17 Citardauq stable form 만 유지 (dd_t compensation 시도 → 효과 없음 revert)
            float a_xy = d.x*d.x + d.y*d.y;
            float b_xy = 2.0f * (q.x*d.x + q.y*d.y);
            float c_xy_o = q.x*q.x + q.y*q.y - R*R;
            float tro_in = -1e20f, tro_out = 1e20f;
            bool  outerSideHits = true;
            if (a_xy < 1e-20f) {
                // axis-parallel ray
                if (c_xy_o > 0.0f) {
                    outerSideHits = false;
                } else {
                    tro_in  = -1e20f; tro_out = 1e20f;
                }
            } else {
                float disc_o = b_xy*b_xy - 4.0f*a_xy*c_xy_o;
                if (disc_o < 0.0f) {
                    outerSideHits = false;
                } else {
                    float sqDisc_o = sqrt(disc_o);
                    // Citardauq stable: q_st = -0.5*(b + sign(b)*√D), t1 = q_st/a, t2 = c/q_st
                    float bsign = (b_xy >= 0.0f) ? 1.0f : -1.0f;
                    float q_st = -0.5f * (b_xy + bsign * sqDisc_o);
                    float t1 = q_st / a_xy;
                    float t2 = (abs(q_st) > 1e-30f) ? (c_xy_o / q_st) : t1;
                    tro_in  = min(t1, t2);
                    tro_out = max(t1, t2);
                }
            }

            // If ray entirely misses outer cylinder, can also miss z-slab → skip
            if (!outerSideHits) continue;

            // --- Inner side quadratic (Rmin>0 일 때만) ---
            // tri_in/tri_out: inner cyl 안으로 들어가는/나가는 시각
            float tri_in = 1e20f, tri_out = -1e20f;  // 기본: 안 만남 (interval 빈 set)
            bool innerSideHits = false;
            if (hasInner && a_xy > 1e-20f) {
                float c_xy_i = q.x*q.x + q.y*q.y - Rmin*Rmin;
                float disc_i = b_xy*b_xy - 4.0f*a_xy*c_xy_i;
                if (disc_i >= 0.0f) {
                    float sqDisc_i = sqrt(disc_i);
                    float bsign_i = (b_xy >= 0.0f) ? 1.0f : -1.0f;
                    float q_st_i = -0.5f * (b_xy + bsign_i * sqDisc_i);
                    float t1_i = q_st_i / a_xy;
                    float t2_i = (abs(q_st_i) > 1e-30f) ? (c_xy_i / q_st_i) : t1_i;
                    tri_in  = min(t1_i, t2_i);
                    tri_out = max(t1_i, t2_i);
                    innerSideHits = true;
                }
            }

            // (광자 위치 분류는 candidate validation 시 hp 좌표로 직접 수행)

            // --- Candidate t 수집 + 표면 위치 valid 검증 ---
            // Hit type code: 0 = outer-side, 1 = inner-side, 2 = z-plane
            float tHit = 1e20f;
            int   hitType = -1;

            // helper lambdas as inlined macros: hit point xy radial² and z
            // (Metal 은 lambda 지원하나 readability 위해 inline 반복)
            float candidates[6];
            int   types[6];
            int   nCand = 0;
            // outer side both intersections
            candidates[nCand] = tro_in;  types[nCand] = 0; nCand++;
            candidates[nCand] = tro_out; types[nCand] = 0; nCand++;
            // inner side
            if (innerSideHits) {
                candidates[nCand] = tri_in;  types[nCand] = 1; nCand++;
                candidates[nCand] = tri_out; types[nCand] = 1; nCand++;
            }
            // z planes (top and bottom 각각)
            if (abs(d.z) > 1e-20f) {
                float invd = 1.0f / d.z;
                candidates[nCand] = (-HL - q.z) * invd; types[nCand] = 2; nCand++;
                candidates[nCand] = ( HL - q.z) * invd; types[nCand] = 2; nCand++;
            }

            for (int i = 0; i < nCand; i++) {
                float t = candidates[i];
                if (t < kSurfTol) continue;
                if (t >= tHit)   continue;

                float3 hp = q + t * d;
                float  r2h = hp.x*hp.x + hp.y*hp.y;
                bool   zInSlab = (abs(hp.z) < HL + kSurfTol);

                bool validHit = false;
                if (types[i] == 0) {
                    // outer side: hit point on |r|=Rmax surface inside z-slab
                    // AND outside inner radius (즉 shell 표면 위)
                    if (zInSlab && (!hasInner || r2h > Rmin*Rmin - 2.0f*Rmin*kSurfTol)) {
                        validHit = true;
                    }
                } else if (types[i] == 1) {
                    // inner side: hit point on |r|=Rmin surface inside z-slab
                    if (zInSlab) validHit = true;
                } else {
                    // z plane: hit point with Rmin ≤ r ≤ Rmax (즉 shell annulus 위)
                    // 2026-06-12 FIX: endCapOpen(열린 끝면=통로) cylinder 의 z-plane 은
                    // 후보 단계에서 제외. 이전에는 valid 후보로 선택된 뒤 아래(구 1751)
                    // `continue` 가 실린더 전체를 버려, 같은 ray 의 더 먼 t 에 있던 정당한
                    // r=R 측벽 교차까지 소거 → 비스듬한 hole 광자가 측벽(n 경계)을
                    // 무이벤트 관통 (sim_light 큰센서 GPU −3%, mid-ring −8%/outer +5.45%
                    // 재분배의 root cause). G4 등가: navigator 가 Air|Air cap 을 no-op
                    // 통과한 뒤 G4Tubs::DistanceToOut 이 측벽을 잡음.
                    if (cyl.endCapOpen == 0u) {
                        bool inAnnulus = (r2h < R*R + 2.0f*R*kSurfTol)
                                      && (!hasInner || r2h > Rmin*Rmin - 2.0f*Rmin*kSurfTol);
                        if (inAnnulus) validHit = true;
                    }
                }
                if (validHit) {
                    tHit = t;
                    hitType = types[i];
                }
            }

            if (hitType < 0) continue;

            // Position 정합성: in-shell 광자는 in-shell exit time 이 가장 큰 candidate 와 같아야
            // (over-stepping 방지). 단 candidate loop 가 이미 smallest valid 만 채택했으므로 OK.

            if (ShouldUpdateClosestHit(tHit, cyl.surfaceId, distToBoundary, closestSurfId)) {
                // Determine normal
                float3 hp  = q + tHit * d;
                float3 rel = hp;
                bool isZPlane = (hitType == 2);
                // (2026-06-12: 구 "endCapOpen && isZPlane → continue" 제거 — 후보 단계
                //  제외로 이동. 여기서 continue 하면 측벽 교차까지 함께 소거되는 버그였음.)
                float3 n;
                if (isZPlane) {
                    n = float3(0, 0, (rel.z > 0) ? 1.0f : -1.0f);
                } else if (hitType == 0) {
                    // Outer side: outward radial (away from axis)
                    float invR = 1.0f / R;
                    n = float3(rel.x * invR, rel.y * invR, 0.0f);
                } else {
                    // Inner side: G4 outward normal from cylinder material side =
                    // radially INWARD (away from shell material, toward hole).
                    float invRm = 1.0f / Rmin;
                    n = float3(-rel.x * invRm, -rel.y * invRm, 0.0f);
                }
                distToBoundary = tHit;
                closestTriIdx = -1;
                closestNormal = n;
                boxHitFlag = false;
                sphereHitFlag = false;
                cylHitFlag = true;
                cylHitIdx = (int)c;
                closestSurfId = cyl.surfaceId;
                cylHitIsZPlane = isZPlane;
            }
        }

        // ========================================================
        // Phase 3.5 (2026-05-05, generic axis): G4Torus SDF Sphere Tracing.
        // Torus axis 는 임의 방향. world → local 회전 후 SDF + Newton, 그 후
        // local normal → world. 광원/렌즈/토러스 자유 배치 가능.
        //   Local: axis = (0,0,1), sdTorus(p) = length(float2(length(p.xy) - RT, p.z)) - Rm
        //   World transform: world coord 의 axis 와 직교 basis (e1, e2, axis)
        // ========================================================
        bool torHitFlag = false;
        int  torHitIdx  = -1;
        for (uint tg = 0; tg < cylTorPolyCounts.y; tg++) {
            TorusGeometry tor = torusGeometries[tg];
            float3 ctr  = float3(tor.center);
            float3 axis = normalize(float3(tor.axis));
            float  RT   = tor.rTor;
            float  Rm   = tor.rMax;

            // Orthonormal basis (e1, e2, axis): cross with (0,0,1) or (1,0,0).
            float3 e1;
            if (abs(axis.z) < 0.9f) {
                e1 = normalize(float3(axis.y, -axis.x, 0.0f));
            } else {
                e1 = normalize(float3(0.0f, axis.z, -axis.y));
            }
            float3 e2 = cross(axis, e1);

            // World ray → local frame
            float3 q_world = float3(photon.position) - ctr;
            float3 d_world = float3(photon.direction);
            float3 P_loc = float3(dot(q_world, e1), dot(q_world, e2), dot(q_world, axis));
            float3 d_loc = float3(dot(d_world, e1), dot(d_world, e2), dot(d_world, axis));

            // Bounding sphere quick-reject (frame invariant)
            float bsR = RT + Rm + 0.01f;
            float qDotD = dot(q_world, d_world);
            float qDotQ = dot(q_world, q_world);
            float bs_disc = qDotD * qDotD - (qDotQ - bsR * bsR);
            if (bs_disc < 0.0f) {
                #if ENABLE_DIAGNOSTICS
                atomic_fetch_add_explicit(&diagCounters[150], 1u, memory_order_relaxed);  // bs_reject
                #endif
                continue;
            }
            #if ENABLE_DIAGNOSTICS
            atomic_fetch_add_explicit(&diagCounters[151], 1u, memory_order_relaxed);  // bs_pass
            #endif

            const float kSurfTol = 1e-4f;

            // Ray-torus intersection (2026-06-01 정석 fix):
            //   sphere-tracing march 로 부호변화 bracket 을 잡고, 정확한 음함수
            //   g(t)=√((√(x²+y²)−RT)²+z²)−Rm 에 bisection 으로 정밀화.
            //   이전 SDF march(tube-circle 근사거리)는 표면을 +1.5cm overshoot →
            //   흡수 torus over-absorption (G/C 0.22). dd-Ferrari quartic 은 fp32
            //   bracket 불안정으로 내부 exit 34% miss. bracket-bisection 이 둘 다 대체:
            //   흡수 G/C 0.22→0.99 (chord sub-mm), 반사 G/C 1.0000 회귀 없음.
            // Phase 3.5 (2026-05-07): Sphere tracing + min-sd hybrid 정석.
            // 이전: kMarchEps=1e-6 → tangent ray 미수렴 30% (mesh reflect 절반 손실).
            // Fix: marching 중 min_sd 추적. iter limit 도달 시 min_sd<1e-3mm 면
            // tangent-actually-hit 로 받음 (1 micron 이하 surface 근접 = optical
            // photon scale 에서 hit). 이전 fix kMarchEps=1e-3 은 reflect 후
            // self-detect 무한 loop 만들어서 NG.
            float tHit = 1e20f;
            {
                float radial0 = sqrt(P_loc.x*P_loc.x + P_loc.y*P_loc.y);
                float ring0 = radial0 - RT;
                float sd0 = sqrt(ring0*ring0 + P_loc.z*P_loc.z) - Rm;
                float startSign = (sd0 < 0.0f) ? -1.0f : 1.0f;
                float t = kSurfTol;
                const int kMaxMarchIter = 256;
                const float kMarchEps = 1e-6f;
                const float kTangentTol = 5e-4f;  // 0.5 micron — tangent-as-hit threshold (1e-3 → 5e-4 false-positive 감소)
                float minSdSeen = 1e20f;
                float tAtMinSd = 0.0f;
                // 2026-06-01 정석 fix: sphere-tracing 은 근사 torus distance(=tube-circle SDF)로
                //   stepping → 표면을 +0.7cm overshoot (흡수 torus over-absorption root cause).
                //   대신 march 로 부호변화 bracket [last-startSign, first-opposite] 을 잡고,
                //   정확한 음함수 g(t)=√(ring²+z²)−Rm 에 bisection → 정확 교차 (overshoot 0).
                //   quartic(dd/bracket 불안정 34% miss) 및 Newton(근사 overshoot 보정) 둘 다 대체.
                float tPrev = kSurfTol;     // 직전 sample (아직 startSign 유지)
                bool  bracketed = false;
                float tBrLo = 0.0f, tBrHi = 0.0f;
                for (int iter = 0; iter < kMaxMarchIter; iter++) {
                    float3 p = P_loc + t * d_loc;
                    float radial = sqrt(p.x*p.x + p.y*p.y);
                    float ring = radial - RT;
                    float sd = sqrt(ring*ring + p.z*p.z) - Rm;
                    float sdSigned = sd * startSign;
                    if (abs(sdSigned) < minSdSeen) { minSdSeen = abs(sdSigned); tAtMinSd = t; }
                    if (sdSigned < kMarchEps) {
                        tBrLo = tPrev; tBrHi = t; bracketed = true;
                        tHit = t;
                        break;
                    }
                    tPrev = t;
                    t += abs(sd);
                    if (t > 2.0f * (RT + Rm) + length(P_loc) + 1.0f) break;
                }
                // bracket [tBrLo(startSign), tBrHi(opposite)] 안에서 정확 음함수 bisection.
                if (bracketed && tBrHi > tBrLo + 1e-9f) {
                    float a = tBrLo, b = tBrHi;
                    for (int bi = 0; bi < 40; bi++) {
                        float tm = 0.5f * (a + b);
                        float3 pm = P_loc + tm * d_loc;
                        float rm = sqrt(pm.x*pm.x + pm.y*pm.y);
                        float rgm = rm - RT;
                        float gm = sqrt(rgm*rgm + pm.z*pm.z) - Rm;
                        if (gm * startSign > 0.0f) a = tm; else b = tm;
                        if (b - a < 1e-7f * (abs(tm) + 1.0f)) break;
                    }
                    tHit = 0.5f * (a + b);
                }
                // Hit 못 찾았지만 minSd 충분히 가까우면 tangent hit (Wächter-Binder 의도).
                // Grazing 차단 (2026-05-07): photon dir 가 surface normal 과 거의 수직 (= surface
                // 평행) 이면 hit 으로 인정하지 않음. 평행 진입 후 ε push 의 false re-hit 으로
                // 무한 reflect loop 만들어지는 것 방지 (axis-aligned mirror cosI≈0 케이스).
                if (tHit >= 1e20f && minSdSeen < kTangentTol && tAtMinSd > kSurfTol) {
                    // approximate normal at tAtMinSd
                    float3 p_min = P_loc + tAtMinSd * d_loc;
                    float radial_min = sqrt(p_min.x*p_min.x + p_min.y*p_min.y);
                    if (radial_min > 1e-10f) {
                        float ring_min = radial_min - RT;
                        float3 n_loc_min = float3(ring_min * p_min.x / radial_min,
                                                  ring_min * p_min.y / radial_min,
                                                  p_min.z);
                        float nlen_min = length(n_loc_min);
                        if (nlen_min > 1e-10f) {
                            n_loc_min /= nlen_min;
                            // |dir·normal| ≈ |cosI|. < 0.05 = 87° 이상 grazing → skip.
                            if (abs(dot(d_loc, n_loc_min)) > 0.05f) {
                                tHit = tAtMinSd;
                            }
                        }
                    }
                }
                #if ENABLE_DIAGNOSTICS
                if (tHit >= 1e20f)
                    atomic_fetch_add_explicit(&diagCounters[152], 1u, memory_order_relaxed);  // no_hit
                else
                    atomic_fetch_add_explicit(&diagCounters[154], 1u, memory_order_relaxed);  // hit_found
                #endif
                // 2026-06-01: bracket-bisection(위)이 정확 교차를 줌 → quartic override 및
                //   Newton refinement 제거. (quartic 은 dd/bracket fp32 불안정으로 내부 exit 34% miss,
                //   Newton 은 근사 SDF overshoot 를 못 잡음. bisection-on-exact-g 가 정석.)
                if (tHit >= 1e20f) continue;
            }

            if (ShouldUpdateClosestHit(tHit, tor.surfaceId, distToBoundary, closestSurfId)) {
                // Normal at hit point (fp32)
                float3 p_hit = P_loc + tHit * d_loc;
                float radial_h = sqrt(p_hit.x*p_hit.x + p_hit.y*p_hit.y);
                float3 n_loc;
                if (radial_h > 1e-10f) {
                    float ring_h = radial_h - RT;
                    n_loc = float3(ring_h * p_hit.x / radial_h,
                                   ring_h * p_hit.y / radial_h,
                                   p_hit.z);
                } else {
                    n_loc = float3(0.0f, 0.0f, sign(p_hit.z));
                }
                float nlen = length(n_loc);
                if (nlen > 1e-10f) n_loc /= nlen;
                float3 n = e1 * n_loc.x + e2 * n_loc.y + axis * n_loc.z;

                distToBoundary = tHit;
                closestTriIdx = -1;
                closestNormal = n;
                boxHitFlag = false;
                sphereHitFlag = false;
                cylHitFlag = false;
                torHitFlag = true;
                torHitIdx = (int)tg;
                closestSurfId = tor.surfaceId;
            }
        }

        // ========================================================
        // Phase 4 (2026-05-18): G4Polyhedra (general N-sided) analytic intersection.
        // Slab approach: N lateral half-spaces + 2 cap half-spaces.
        // Each lateral side i: angle = phiStart + (i+0.5)*(phiTotal/N)
        //   outward normal (local XY) = (cos a, sin a, 0)
        //   plane offset (apothem) = RMax * cos(π/N) for full phi
        // Local frame: axis = local Z, orthogonal e1, e2 in local XY.
        // Currently supports: full phi (phiTotal=2π) only. RMin=0 (solid).
        // ========================================================
        bool polyHitFlag = false;
        int  polyHitIdx  = -1;
        for (uint pg = 0; pg < cylTorPolyCounts.z; pg++) {
            PolyhedraGeometry ph = polyhedraGeometries[pg];
            float3 ctr = float3(ph.centerX, ph.centerY, ph.centerZ);
            float3 ax = normalize(float3(ph.axisX, ph.axisY, ph.axisZ));
            uint N = ph.numSides;
            if (N < 3) continue;

            // Local frame (e1, e2, ax) supplied by host (worldTransform).
            // PhiStart/face-midpoint angles 는 이 (e1, e2) 평면 안에서 정의.
            float3 e1 = normalize(float3(ph.e1X, ph.e1Y, ph.e1Z));
            float3 e2 = normalize(float3(ph.e2X, ph.e2Y, ph.e2Z));

            // World ray → local frame
            float3 q_world = float3(photon.position) - ctr;
            float3 d_world = float3(photon.direction);
            float3 P_loc = float3(dot(q_world, e1), dot(q_world, e2), dot(q_world, ax));
            float3 d_loc = float3(dot(d_world, e1), dot(d_world, e2), dot(d_world, ax));

            // Slab intersection: tEnter = max(entry t's), tExit = min(exit t's)
            // 2026-05-19 정석 fix: entry/exit normal 둘 다 main loop 에서 직접 track.
            // 이전 코드는 exit normal 을 fp32 tolerance (`abs(t - tExit) < 1e-6`) 매칭으로
            // recompute. 광자가 정확히 face 위에 있어 tExit≈0, 또는 multiple face 가
            // 거의 같은 t 면 wrong face match → closestNormal = wrong face → multi-bounce
            // 시 광자 stuck → STACK_OVERFLOW 100%.
            float tEnter = -1e20f;
            float tExit  =  1e20f;
            float3 normalEnterLoc = float3(0,0,0);
            float3 normalExitLoc  = float3(0,0,0);
            bool entryFromCap = false;
            bool exitFromCap  = false;

            // 2 axial cap planes (local Z = ±halfLengthAxis)
            // 2026-05-19 fp32 stability fix: parallel threshold 1e-10 → 1e-6.
            // fp32 dot product 의 cumulative error ~O(3 ULP) = 3e-7. 1e-10 가 underflow
            // region 안 → denormal/instability. iso5n 1.620/1.625/1.630 의 erratic G/C
            // 가 axis ⊥ beam 시 d_loc.z=6e-17 (fp32 noise 영역) 에서 발생.
            float hl = ph.halfLengthAxis;
            if (abs(d_loc.z) > 1e-6f) {
                float tz_lo = (-hl - P_loc.z) / d_loc.z;
                float tz_hi = ( hl - P_loc.z) / d_loc.z;
                float te_z = min(tz_lo, tz_hi);
                float tx_z = max(tz_lo, tz_hi);
                // Z entry cap: d_loc.z>0 면 lower cap (-hl) 가 entry, normal=(0,0,-1)
                //              d_loc.z<0 면 upper cap (+hl) 가 entry, normal=(0,0,+1)
                if (te_z > tEnter) {
                    tEnter = te_z;
                    normalEnterLoc = (d_loc.z > 0) ? float3(0,0,-1) : float3(0,0,1);
                    entryFromCap = true;
                }
                // Z exit cap: 반대 면
                if (tx_z < tExit) {
                    tExit = tx_z;
                    normalExitLoc = (d_loc.z > 0) ? float3(0,0,1) : float3(0,0,-1);
                    exitFromCap = true;
                }
            } else {
                // ray parallel to cap planes — outside if |P_loc.z| > hl
                if (abs(P_loc.z) > hl) continue;
            }

            // N lateral half-spaces
            float angleStep = ph.phiTotal / (float)N;
            float apothem = ph.rMax * cos(M_PI_F / (float)N);
            bool reject = false;
            for (uint k = 0; k < N; k++) {
                float ang = ph.phiStart + ((float)k + 0.5f) * angleStep;
                float3 n_loc = float3(cos(ang), sin(ang), 0.0f);
                float numer = apothem - dot(P_loc, n_loc);
                float denom = dot(d_loc, n_loc);
                if (abs(denom) < 1e-6f) {
                    if (numer < 0.0f) { reject = true; break; }
                    continue;
                }
                float t = numer / denom;
                if (denom > 0.0f) {
                    if (t < tExit) {
                        tExit = t;
                        normalExitLoc = n_loc;
                        exitFromCap = false;
                    }
                } else {
                    if (t > tEnter) {
                        tEnter = t;
                        normalEnterLoc = n_loc;
                        entryFromCap = false;
                    }
                }
            }
            if (reject) continue;
            if (tEnter > tExit || tExit < 1e-6f) continue;

            // Choose nearest forward intersection
            const float kSurfTol = 1e-4f;
            float tHit;
            float3 nLoc;
            if (tEnter > kSurfTol) {
                tHit = tEnter;
                nLoc = normalEnterLoc;
            } else {
                tHit = tExit;
                nLoc = normalExitLoc;
            }
            if (tHit <= kSurfTol || !ShouldUpdateClosestHit(tHit, ph.surfaceId, distToBoundary, closestSurfId)) continue;

            float3 n_world = e1 * nLoc.x + e2 * nLoc.y + ax * nLoc.z;
            n_world = normalize(n_world);

            distToBoundary = tHit;
            closestTriIdx = -1;
            closestNormal = n_world;
            boxHitFlag = false;
            sphereHitFlag = false;
            cylHitFlag = false;
            torHitFlag = false;
            polyHitFlag = true;
            polyHitIdx = (int)pg;
            closestSurfId = ph.surfaceId;
        }

        // 최단 거리 프로세스 결정
        float stepDist = min(min(min(distToAbsorb, distToRayleigh),
                                 min(distToMie, distToWLS)),
                             distToBoundary);

        // 월드 경계 — parallel-world scoring 정석 fix (2026-04-26):
        // photon이 mass geometry boundary 없이 World 밖으로 직진하는 step
        // (예: reflected ray가 Glass mass volume을 안 만나는 경우) 에서
        // stepDist를 World exit point까지 clamp 한다. 그렇게 하지 않으면
        // newPos가 World 밖으로 튕겨나가 OUT_OF_WORLD 즉시 break → 그 step
        // 의 transit-hit (parallel-world fluence scoring) 코드가 실행되지
        // 않아 reflected/grazing 광자가 PW ScoreBox에서 누락된다.
        bool willExitWorld = false;
        {
            float3 worldHalf = float3(config.worldSizeX, config.worldSizeY,
                                     config.worldSizeZ) * 0.5f;
            float3 pos = float3(photon.position);
            float3 dir = float3(photon.direction);
            float tExit = 1e20f;
            if (abs(dir.x) > 1e-20f) {
                float t1 = (-worldHalf.x - pos.x) / dir.x;
                float t2 = ( worldHalf.x - pos.x) / dir.x;
                tExit = min(tExit, max(t1, t2));
            }
            if (abs(dir.y) > 1e-20f) {
                float t1 = (-worldHalf.y - pos.y) / dir.y;
                float t2 = ( worldHalf.y - pos.y) / dir.y;
                tExit = min(tExit, max(t1, t2));
            }
            if (abs(dir.z) > 1e-20f) {
                float t1 = (-worldHalf.z - pos.z) / dir.z;
                float t2 = ( worldHalf.z - pos.z) / dir.z;
                tExit = min(tExit, max(t1, t2));
            }
            tExit = max(0.0f, tExit);
            if (tExit < stepDist) {
                stepDist = tExit;
                willExitWorld = true;
                // boundary 분기 진입 차단: 모든 hit flag reset
                closestTriIdx = -1;
                boxHitFlag = false;
                sphereHitFlag = false;
                cylHitFlag = false;
                torHitFlag = false;
                polyHitFlag = false;
            }
        }

        // 2026-05-17 propagation step FMA: pos + stepDist*dir 의 fp32 두 round 를
        // single-FMA round 으로 → ULP 절반 감소. Metal compiler 가 multi-line 표현식
        // 에서는 FMA contraction 안 적용하므로 명시적 fma() 호출.
        float3 newPos = float3(
            fma(stepDist, photon.direction.x, photon.position.x),
            fma(stepDist, photon.direction.y, photon.position.y),
            fma(stepDist, photon.direction.z, photon.position.z)
        );

        // 2026-05-19 정석 fix: polyhedra hit 시 newPos 를 face plane 위로 snap.
        // fp32 cumulative round error 차단 — newPos · n_outward = exact_offset 강제.
        if (polyHitFlag && polyHitIdx >= 0) {
            PolyhedraGeometry ph_snap = polyhedraGeometries[polyHitIdx];
            float3 ctr_snap = float3(ph_snap.centerX, ph_snap.centerY, ph_snap.centerZ);
            float3 ax_snap = normalize(float3(ph_snap.axisX, ph_snap.axisY, ph_snap.axisZ));
            float3 nW = closestNormal;
            // exact offset: cap 의 경우 hl, lateral 의 경우 apothem.
            float dot_ax = abs(dot(nW, ax_snap));
            float exact_offset;
            if (dot_ax > 0.999f) {
                exact_offset = ph_snap.halfLengthAxis;
            } else {
                exact_offset = ph_snap.rMax * cos(M_PI_F / (float)ph_snap.numSides);
            }
            float current_dot = dot(newPos - ctr_snap, nW);
            float shift = exact_offset - current_dot;
            newPos = newPos + shift * nW;
        }

        // 위치 및 시간 업데이트 (Fix 69: MaterialGPU 접근 없이 LUT 직접)
        float n_cached = InterpolateLUT(matLUTs[currentMatId].rindex, photon.energy);
        if (n_cached <= 0.0) n_cached = 1.0;
        float velocity = SPEED_OF_LIGHT / n_cached;

        // Fix 41b + Fix 63: 스코어링 AABB 관통 체크 (빠른 거부 테스트 추가)
        if (scoringAABBEnabled && stepDist > 0) {
            float3 oldPos = float3(photon.position);
            float3 dir = float3(photon.direction);

            // Fix 63: 빠른 거부 — oldPos와 newPos 모두 AABB 밖이면 skip
            // 양 끝점이 같은 쪽에 있으면 교차 불가능
            bool canIntersect = true;
            if ((oldPos.x < scoringMin.x && newPos.x < scoringMin.x) ||
                (oldPos.x > scoringMax.x && newPos.x > scoringMax.x) ||
                (oldPos.y < scoringMin.y && newPos.y < scoringMin.y) ||
                (oldPos.y > scoringMax.y && newPos.y > scoringMax.y) ||
                (oldPos.z < scoringMin.z && newPos.z < scoringMin.z) ||
                (oldPos.z > scoringMax.z && newPos.z > scoringMax.z)) {
                canIntersect = false;
            }

            if (canIntersect) {
            // 광선-AABB 교차: 광선 oldPos + t*dir, t ∈ [0, stepDist]
            float tEnter = 0.0;
            float tExit = stepDist;

            // X축
            if (abs(dir.x) > 1e-10) {
                float t1 = (scoringMin.x - oldPos.x) / dir.x;
                float t2 = (scoringMax.x - oldPos.x) / dir.x;
                if (t1 > t2) { float tmp = t1; t1 = t2; t2 = tmp; }
                tEnter = max(tEnter, t1);
                tExit = min(tExit, t2);
            } else {
                if (oldPos.x < scoringMin.x || oldPos.x > scoringMax.x) {
                    tEnter = tExit + 1.0;
                }
            }

            // Y축
            if (tEnter <= tExit) {
                if (abs(dir.y) > 1e-10) {
                    float t1 = (scoringMin.y - oldPos.y) / dir.y;
                    float t2 = (scoringMax.y - oldPos.y) / dir.y;
                    if (t1 > t2) { float tmp = t1; t1 = t2; t2 = tmp; }
                    tEnter = max(tEnter, t1);
                    tExit = min(tExit, t2);
                } else {
                    if (oldPos.y < scoringMin.y || oldPos.y > scoringMax.y) {
                        tEnter = tExit + 1.0;
                    }
                }
            }

            // Z축
            if (tEnter <= tExit) {
                if (abs(dir.z) > 1e-10) {
                    float t1 = (scoringMin.z - oldPos.z) / dir.z;
                    float t2 = (scoringMax.z - oldPos.z) / dir.z;
                    if (t1 > t2) { float tmp = t1; t1 = t2; t2 = tmp; }
                    tEnter = max(tEnter, t1);
                    tExit = min(tExit, t2);
                } else {
                    if (oldPos.z < scoringMin.z || oldPos.z > scoringMax.z) {
                        tEnter = tExit + 1.0;
                    }
                }
            }

            #if ENABLE_DIAGNOSTICS
            // Fix 60: AABB 스코어링 검증 진단
            {
                if (tEnter < tExit) {
                    uint tLenNm = (uint)((tExit - tEnter) * 1e6);
                    atomic_fetch_add_explicit(&diagCounters[28], tLenNm, memory_order_relaxed);
                    atomic_fetch_add_explicit(&diagCounters[29], 1, memory_order_relaxed);
                }
                float3 np = float3(photon.position) + stepDist * dir;
                bool npInside = (np.x >= scoringMin.x && np.x <= scoringMax.x &&
                                 np.y >= scoringMin.y && np.y <= scoringMax.y &&
                                 np.z >= scoringMin.z && np.z <= scoringMax.z);
                if (npInside) {
                    uint stepNm = (uint)(stepDist * 1e6);
                    atomic_fetch_add_explicit(&diagCounters[30], stepNm, memory_order_relaxed);
                    atomic_fetch_add_explicit(&diagCounters[31], 1, memory_order_relaxed);
                }
            }
            #endif

            // AABB 교차 구간이 존재하면 transit hit 기록
            if (tEnter < tExit) {
                float transitLength = tExit - tEnter;
                float3 entryPoint = oldPos + tEnter * dir;

                uint tIdx = atomic_fetch_add_explicit(&transitCount, 1, memory_order_relaxed);
                if (tIdx < maxTransitHits) {
                    TransitHit th;
                    th.position = packed_float3(entryPoint);
                    th.direction = dir;
                    th.energy = photon.energy;
                    th.weight = transitLength;             // geometric 경로 길이 (mm)
                    th.photonWeight = photon.weight;       // 광자 가중치 (기본 1.0)
                    th.materialId = photon.materialId;     // F2 정석: scorer material 필터용
                    transitHits[tIdx] = th;
                }
            }
            }  // canIntersect
        }

        // SF1 pre-boundary baseline 제거됨 (2026-05-12 cleanup): post-boundary
        // refactor 가 production path. 이전 baseline 코드는 git history 참조.

        photon.position = packed_float3(newPos);
        photon.time += stepDist / velocity;
        photon.stepCount++;

        // =======================================================
        // 승리한 프로세스 실행
        // =======================================================

        if (stepDist == distToBoundary && (closestTriIdx >= 0 || boxHitFlag || sphereHitFlag || cylHitFlag || torHitFlag || polyHitFlag)) {
            // 경계 상호작용
            // Analytic hit (Box/Sphere/Cylinder/Torus) 의 경우 hitTri 를 합성
            TriangleAttr hitTri;
            if (boxHitFlag && boxHitIdx >= 0) {
                BoxGeometry box = boxGeometries[boxHitIdx];
                hitTri.normal            = packed_float3(closestNormal);
                hitTri.volumeId          = box.volumeId;
                hitTri.materialIdInside  = box.materialIdInside;
                // Sibling adjacency: outsideTestPoint 가 다른 등록된 box 안이면
                // 그 box 의 inside material 이 실제 outside material (parent 가 아님).
                // wrap-wrap touching face 등에서 spurious Fresnel 방지.
                {
                    uint resolvedMatOut = box.materialIdOutside;
                    const float kSiblingProbeEps = 1e-3f;
                    float3 outsideTestPoint = float3(photon.position) + kSiblingProbeEps * float3(closestNormal);
                    for (uint b2 = 0; b2 < numBoxGeometries; b2++) {
                        if (b2 == (uint)boxHitIdx) continue;
                        BoxGeometry box2 = boxGeometries[b2];
                        float3 d2 = outsideTestPoint - float3(box2.center);
                        if (abs(d2.x) <= box2.HLX &&
                            abs(d2.y) <= box2.HLY &&
                            abs(d2.z) <= box2.HLZ) {
                            resolvedMatOut = box2.materialIdInside;
                            break;
                        }
                    }
                    hitTri.materialIdOutside = resolvedMatOut;
                }
                hitTri.surfaceId         = box.surfaceId;
                hitTri._pad              = 0;
            } else if (sphereHitFlag && sphereHitIdx >= 0) {
                SphereGeometry sph = sphereGeometries[sphereHitIdx];
                hitTri.normal            = packed_float3(closestNormal);  // outward
                hitTri.volumeId          = sph.volumeId;
                hitTri.materialIdInside  = sph.materialIdInside;
                hitTri.materialIdOutside = sph.materialIdOutside;
                hitTri.surfaceId         = sph.surfaceId;
                hitTri._pad              = 0;
            } else if (cylHitFlag && cylHitIdx >= 0) {
                CylinderGeometry cyl = cylinderGeometries[cylHitIdx];
                hitTri.normal            = packed_float3(closestNormal);
                hitTri.volumeId          = cyl.volumeId;
                hitTri.materialIdInside  = cyl.materialIdInside;
                // Sibling adjacency (box 와 동일, 2026-05-21): cylinder 측면 hit 시
                // outsideTestPoint 가 다른 cylinder 안이면 그 cylinder 의 inside
                // material 이 실제 outside material (parent 가 아님). FiberCore(parent
                // World)→FiberClad(sibling) 처럼 core→parent(Air) 오판 시 spurious
                // core→Air TIR 로 cladding-guided 광이 통째로 갇히는 것 방지.
                {
                    uint resolvedMatOut = cyl.materialIdOutside;
                    const float kSiblingProbeEps = 1e-3f;
                    float3 outsideTestPoint = float3(photon.position) + kSiblingProbeEps * float3(closestNormal);
                    for (uint c2 = 0; c2 < cylTorPolyCounts.x; c2++) {
                        if (c2 == (uint)cylHitIdx) continue;
                        CylinderGeometry cyl2 = cylinderGeometries[c2];
                        float3 d2 = outsideTestPoint - float3(cyl2.center);
                        float r2 = d2.x*d2.x + d2.y*d2.y;
                        if (r2 <= cyl2.radius*cyl2.radius
                            && r2 >= cyl2.innerRadius*cyl2.innerRadius
                            && abs(d2.z) <= cyl2.halfLength) {
                            resolvedMatOut = cyl2.materialIdInside;
                            break;
                        }
                    }
                    hitTri.materialIdOutside = resolvedMatOut;
                }
                hitTri.surfaceId         = cyl.surfaceId;
                hitTri._pad              = 0;
            } else if (torHitFlag && torHitIdx >= 0) {
                TorusGeometry tor = torusGeometries[torHitIdx];
                hitTri.normal            = packed_float3(closestNormal);
                hitTri.volumeId          = tor.volumeId;
                hitTri.materialIdInside  = tor.materialIdInside;
                hitTri.materialIdOutside = tor.materialIdOutside;
                hitTri.surfaceId         = tor.surfaceId;
                hitTri._pad              = 0;
            } else if (polyHitFlag && polyHitIdx >= 0) {
                // Phase 4 (2026-05-18): G4Polyhedra hit
                PolyhedraGeometry ph = polyhedraGeometries[polyHitIdx];
                hitTri.normal            = packed_float3(closestNormal);
                hitTri.volumeId          = ph.volumeId;
                hitTri.materialIdInside  = ph.materialIdInside;
                hitTri.materialIdOutside = ph.materialIdOutside;
                hitTri.surfaceId         = ph.surfaceId;
                hitTri._pad              = 0;
            } else {
                hitTri = triangles[closestTriIdx];  // Fix 71: 32B
                // 2026-05-22: mesh tri runtime sibling adjacency (box analytic
                // line 2604-2622 와 동일 원리). World-child wrap slab(RefX/Y/Z 등)이
                // box-analytic 결정과 coincident 면, slab mesh tri 의 matIn/matOut 가
                // parent(World Air)/Air 로 남아 광자 currentMatId(=결정 material)와
                // 불일치 → matIdOut 결정(아래 2713-2722)에서 continue → boundary
                // (reflector) 통째로 skip. Fix 82 mesh-mesh sibling adjacency 는
                // mesh↔box-analytic coincident 를 못 잡으므로, hit point 양쪽을
                // 등록된 box geometry 로 runtime 조회해 matIn/matOut 를 정정한다.
                // (currentMatId 가 이미 일치하는 정상 mesh 면 분기 진입 안 함 → 무영향)
                if (DISABLE_MESH_SIBLING_TEST == 0 &&
                    currentMatId != hitTri.materialIdInside &&
                    currentMatId != hitTri.materialIdOutside) {
                    const float kMeshSiblingEps = 1e-3f;
                    float3 inPt  = float3(photon.position) - kMeshSiblingEps * closestNormal;
                    float3 outPt = float3(photon.position) + kMeshSiblingEps * closestNormal;
                    // box (axis-aligned)
                    for (uint b2 = 0; b2 < numBoxGeometries; b2++) {
                        BoxGeometry box2 = boxGeometries[b2];
                        float3 di = inPt - float3(box2.center);
                        if (abs(di.x) <= box2.HLX && abs(di.y) <= box2.HLY && abs(di.z) <= box2.HLZ)
                            hitTri.materialIdInside = box2.materialIdInside;
                        float3 doo = outPt - float3(box2.center);
                        if (abs(doo.x) <= box2.HLX && abs(doo.y) <= box2.HLY && abs(doo.z) <= box2.HLZ)
                            hitTri.materialIdOutside = box2.materialIdInside;
                    }
                    // sphere (full/shell RMin~RMax; theta-partial 은 보수적으로 무시)
                    for (uint s2 = 0; s2 < cylTorPolyCounts.w; s2++) {
                        SphereGeometry sph2 = sphereGeometries[s2];
                        float3 di = inPt - float3(sph2.center); float ri2 = dot(di, di);
                        if (ri2 <= sph2.radius*sph2.radius && ri2 >= sph2.innerRadius*sph2.innerRadius)
                            hitTri.materialIdInside = sph2.materialIdInside;
                        float3 doo = outPt - float3(sph2.center); float ro2 = dot(doo, doo);
                        if (ro2 <= sph2.radius*sph2.radius && ro2 >= sph2.innerRadius*sph2.innerRadius)
                            hitTri.materialIdOutside = sph2.materialIdInside;
                    }
                    // cylinder (Z-aligned, tube/shell)
                    for (uint c2 = 0; c2 < cylTorPolyCounts.x; c2++) {
                        CylinderGeometry cyl2 = cylinderGeometries[c2];
                        float3 di = inPt - float3(cyl2.center); float ri2 = di.x*di.x + di.y*di.y;
                        if (ri2 <= cyl2.radius*cyl2.radius && ri2 >= cyl2.innerRadius*cyl2.innerRadius && abs(di.z) <= cyl2.halfLength)
                            hitTri.materialIdInside = cyl2.materialIdInside;
                        float3 doo = outPt - float3(cyl2.center); float ro2 = doo.x*doo.x + doo.y*doo.y;
                        if (ro2 <= cyl2.radius*cyl2.radius && ro2 >= cyl2.innerRadius*cyl2.innerRadius && abs(doo.z) <= cyl2.halfLength)
                            hitTri.materialIdOutside = cyl2.materialIdInside;
                    }
                    // torus (generic axis)
                    for (uint t2 = 0; t2 < cylTorPolyCounts.y; t2++) {
                        TorusGeometry tor2 = torusGeometries[t2];
                        float3 ax = normalize(float3(tor2.axis));
                        float3 di = inPt - float3(tor2.center); float zi = dot(di, ax);
                        float rhoi = length(di - zi*ax); float dti = sqrt((rhoi - tor2.rTor)*(rhoi - tor2.rTor) + zi*zi);
                        if (dti <= tor2.rMax) hitTri.materialIdInside = tor2.materialIdInside;
                        float3 doo = outPt - float3(tor2.center); float zo = dot(doo, ax);
                        float rhoo = length(doo - zo*ax); float dto = sqrt((rhoo - tor2.rTor)*(rhoo - tor2.rTor) + zo*zo);
                        if (dto <= tor2.rMax) hitTri.materialIdOutside = tor2.materialIdInside;
                    }
                    // polyhedra (N-sided, full phi, solid)
                    for (uint p2 = 0; p2 < cylTorPolyCounts.z; p2++) {
                        PolyhedraGeometry ph2 = polyhedraGeometries[p2];
                        if (PointInPolyhedra(inPt,  ph2)) hitTri.materialIdInside  = ph2.materialIdInside;
                        if (PointInPolyhedra(outPt, ph2)) hitTri.materialIdOutside = ph2.materialIdInside;
                    }
                }
            }

            #if ENABLE_DIAGNOSTICS
            // === HW RT Intersection Precision 진단 ===
            {
                float3 v0 = float3(0); // Fix 71: v0 removed from TriangleAttr (진단 비활성)
                float3 triN = float3(hitTri.normal);
                float distToPlane = abs(dot(newPos - v0, triN));
                atomic_fetch_add_explicit(&diagCounters[16], 1, memory_order_relaxed);
                uint distUint = as_type<uint>(distToPlane);
                atomic_fetch_max_explicit(&diagCounters[17], distUint, memory_order_relaxed);
                uint distNm = (uint)(distToPlane * 1e6);
                atomic_fetch_add_explicit(&diagCounters[18], distNm, memory_order_relaxed);
                if (distToPlane > 0.001)
                    atomic_fetch_add_explicit(&diagCounters[19], 1, memory_order_relaxed);
                if (distToPlane > 0.01)
                    atomic_fetch_add_explicit(&diagCounters[20], 1, memory_order_relaxed);
            }
            #endif

            // M9 수정: volumeId 업데이트 — 법선 방향 기반 판별 개선
            uint matIdIn = currentMatId;
            uint matIdOut;
            uint newVolumeId = hitTri.volumeId;

            // Fix (2026-04-23, Expert F): currentMatId 일치 기반 matIdOut 결정
            // dirDotN 기반은 reflect/TIR 후 dir 반전 시 같은 face backward
            // side 를 hit 했을 때 잘못된 matIdOut → spurious Fresnel → multi-
            // bounce path 부풀림. currentMatId 일치 보장 + Air-in-Air 트리비얼.
            {
                if (currentMatId == hitTri.materialIdInside &&
                    currentMatId == hitTri.materialIdOutside) {
                    matIdOut = currentMatId;
                } else if (currentMatId == hitTri.materialIdInside) {
                    matIdOut = hitTri.materialIdOutside;
                } else if (currentMatId == hitTri.materialIdOutside) {
                    matIdOut = hitTri.materialIdInside;
                } else {
                    continue;
                }
            }

            // Fix: matIdOut 범위 검증 — 잘못된 삼각형 데이터로 버퍼 오버리드 방지
            if (matIdOut >= MAX_MATERIALS) {
                photon.status = ABSORBED;
                break;
            }

            bool hasSurface = (hitTri.surfaceId > 0 && hitTri.surfaceId < MAX_SURFACES);

            #if ENABLE_DIAGNOSTICS
            // Fix 56 진단: matIdIn==matIdOut, 볼륨별 WithSurface 카운터
            if (matIdIn == matIdOut) {
                atomic_fetch_add_explicit(&diagCounters[21], 1, memory_order_relaxed);
                if (hasSurface)
                    atomic_fetch_add_explicit(&diagCounters[22], 1, memory_order_relaxed);
            }
            if (hasSurface) {
                uint vid = hitTri.volumeId;
                if (vid < 4)
                    atomic_fetch_add_explicit(&diagCounters[23 + vid], 1, memory_order_relaxed);
            }
            #endif

            #if ENABLE_DIAGNOSTICS
            uint matBefore = photon.materialId;  // 진단: 경계 전 물질 ID
            // Fix 61: 경계 처리 전 Fresnel R 계산 (궤적 덤프용)
            float traj_n1 = 0, traj_n2 = 0, traj_cosI = 0, traj_R = 0;
            if (dumpTrajectory) {
                traj_n1 = InterpolateProperty(materials[matIdIn].refractiveIndex, photon.energy);
                traj_n2 = InterpolateProperty(materials[matIdOut].refractiveIndex, photon.energy);
                if (traj_n1 <= 0) traj_n1 = 1.0;
                if (traj_n2 <= 0) traj_n2 = 1.0;
                float3 fN = closestNormal;
                traj_cosI = -dot(float3(photon.direction), fN);
                if (traj_cosI < 0) { fN = -fN; traj_cosI = -traj_cosI; }
                float Rs, Rp;
                traj_R = FresnelReflectance(traj_cosI, traj_n1, traj_n2,
                                            float3(photon.direction), fN,
                                            float3(photon.polarization), Rs, Rp);
            }
            #endif

            // Fix 64b: LUT로 n1/n2 사전 계산하여 Fresnel에 전달 (이중 조회 제거)
            // 2026-05-15: bnd_n1 은 matIdIn==currentMatId 이므로 n_cached (위 step에서 조회) 재사용
            //   — 같은 LUT 같은 energy 의 boundary step 당 중복 조회 제거.
            float bnd_n1 = n_cached;
            float bnd_n2 = InterpolateLUT(matLUTs[matIdOut].rindex, photon.energy);

            #if ENABLE_DIAGNOSTICS
            // Expert E (2026-04-22): Glass exit face 분포 측정
            // matIdIn=Glass(n>1.4), matIdOut=Air(n<1.1) 인 경우 face 분류
            // slots 50-53: face별 attempts (50=-Z bottom, 51=+Z top, 52=±X, 53=±Y)
            // slots 54-57: face별 reflected (slot+4)
            // slots 58/59: +X / -X 분리
            bool dbg_isGlassExit = (bnd_n1 > 1.4f && bnd_n2 < 1.1f);
            int dbg_faceSlot = -1;
            if (dbg_isGlassExit) {
                float ax = fabs(closestNormal.x);
                float ay = fabs(closestNormal.y);
                float az = fabs(closestNormal.z);
                if (az >= ax && az >= ay) {
                    dbg_faceSlot = (closestNormal.z > 0.0f) ? 51 : 50;
                } else if (ax >= ay) {
                    dbg_faceSlot = 52;
                    if (closestNormal.x > 0.0f)
                        atomic_fetch_add_explicit(&diagCounters[58], 1, memory_order_relaxed);
                    else
                        atomic_fetch_add_explicit(&diagCounters[59], 1, memory_order_relaxed);
                } else {
                    dbg_faceSlot = 53;
                }
                atomic_fetch_add_explicit(&diagCounters[dbg_faceSlot], 1, memory_order_relaxed);
            }
            // Expert E: reflectedCount 변화 측정용 사전 캡처
            uint dbg_reflBefore = photon.reflectedCount;
            // Expert SideX-TIR z-distribution (2026-04-22): SideX face TIR 광자
            // 의 photon.position.z 분포 측정. ProcessBoundaryFresnel 호출 시
            // reflect 분기에서 normal push (line 365) 가 발생하므로 사전에 캡처.
            // TIR strict 판정: bnd_n1 > bnd_n2 (Glass→Air) 이고 sin²T > 1.
            float dbg_zBefore = float3(photon.position).z;
            bool dbg_isTIR = false;
            if (dbg_isGlassExit && dbg_faceSlot == 52 && bnd_n1 > bnd_n2) {
                float3 dbg_dir = float3(photon.direction);
                float3 dbg_N = closestNormal;
                float dbg_cosI = -dot(dbg_dir, dbg_N);
                if (dbg_cosI < 0.0) dbg_cosI = -dbg_cosI;
                float dbg_ratio = bnd_n1 / bnd_n2;
                float dbg_sinT2 = dbg_ratio * dbg_ratio * (1.0 - dbg_cosI * dbg_cosI);
                dbg_isTIR = (dbg_sinT2 > 1.0);
                // [SideXDiag] TIR/refract 카운트 + cosI bin histogram
                // (slot 60 은 기존 hackRejectCnt 와 conflict 해서 70-90 사용)
                if (dbg_isTIR) {
                    atomic_fetch_add_explicit(&diagCounters[70], 1u, memory_order_relaxed);
                } else {
                    atomic_fetch_add_explicit(&diagCounters[71], 1u, memory_order_relaxed);
                }
                // cosI histogram: 10 bins (0.0-0.1, 0.1-0.2, ..., 0.9-1.0)
                // [H4-followup 2026-04-27] moved 110→1400 to avoid BulkAbsorb slot conflict
                uint cosI_bin = (uint)(dbg_cosI * 10.0f);
                if (cosI_bin >= 10u) cosI_bin = 9u;
                atomic_fetch_add_explicit(&diagCounters[4000u + cosI_bin], 1u, memory_order_relaxed);

                // stepDist log10 histogram (mm). 11 bins:
                //   bin 0: < 1e-9   (StepTooSmall 영역, G4 fCarTolerance)
                //   bin 1: 1e-9 ~ 1e-8
                //   bin 2: 1e-8 ~ 1e-7
                //   bin 3: 1e-7 ~ 1e-6
                //   bin 4: 1e-6 ~ 1e-5
                //   bin 5: 1e-5 ~ 1e-4
                //   bin 6: 1e-4 ~ 1e-3
                //   bin 7: 1e-3 ~ 1e-2
                //   bin 8: 1e-2 ~ 1e-1
                //   bin 9: 1e-1 ~ 1.0
                //   bin 10: > 1.0
                // [H4-followup 2026-04-27] moved 120→1410 to avoid matChange/stepZero slot conflict
                float dbg_step = max(stepDist, 1e-12f);
                int sd_bin = (int)floor(log10(dbg_step)) + 9;  // -9 → bin 0, 0 → bin 9
                if (sd_bin < 0) sd_bin = 0;
                if (sd_bin > 10) sd_bin = 10;
                atomic_fetch_add_explicit(&diagCounters[4010u + (uint)sd_bin], 1u, memory_order_relaxed);
            }
            // Expert SideX-hit-pos (2026-04-22): SideX face hit 위치 정확한 (x,y,z) 첫 100개 dump
            //   조건: Glass 안 + SideX face hit (dbg_faceSlot == 52)
            //   슬롯: dc[100] = atomic counter, dc[200+9*i+0..8] = (x,y,z,dx,dy,dz,nx,ny,nz) raw float bits
            //   Multi-bounce 여부 / TIR 여부도 같이 capture: dc[200+9*i] 의 lower bits 는 그대로 float bit pattern.
            //   추가 메타: slot 1100..1199 = flags(8bit multi-bounce) | (TIR<<8) | (matIdIn<<16) per entry
            if (dbg_isGlassExit && dbg_faceSlot == 52) {
                uint dbg_slot = atomic_fetch_add_explicit(&diagCounters[100], 1u, memory_order_relaxed);
                if (dbg_slot < 100u) {
                    float3 dbg_p = float3(photon.position);
                    float3 dbg_d = float3(photon.direction);
                    float3 dbg_N2 = closestNormal;
                    uint base = 200u + dbg_slot * 9u;
                    atomic_store_explicit(&diagCounters[base + 0u], as_type<uint>(dbg_p.x), memory_order_relaxed);
                    atomic_store_explicit(&diagCounters[base + 1u], as_type<uint>(dbg_p.y), memory_order_relaxed);
                    atomic_store_explicit(&diagCounters[base + 2u], as_type<uint>(dbg_p.z), memory_order_relaxed);
                    atomic_store_explicit(&diagCounters[base + 3u], as_type<uint>(dbg_d.x), memory_order_relaxed);
                    atomic_store_explicit(&diagCounters[base + 4u], as_type<uint>(dbg_d.y), memory_order_relaxed);
                    atomic_store_explicit(&diagCounters[base + 5u], as_type<uint>(dbg_d.z), memory_order_relaxed);
                    atomic_store_explicit(&diagCounters[base + 6u], as_type<uint>(dbg_N2.x), memory_order_relaxed);
                    atomic_store_explicit(&diagCounters[base + 7u], as_type<uint>(dbg_N2.y), memory_order_relaxed);
                    atomic_store_explicit(&diagCounters[base + 8u], as_type<uint>(dbg_N2.z), memory_order_relaxed);
                    uint meta = (photon.flags & 0xFFu)
                              | ((dbg_isTIR ? 1u : 0u) << 8)
                              | ((matIdIn & 0xFFu) << 16)
                              | ((matIdOut & 0xFFu) << 24);
                    atomic_store_explicit(&diagCounters[1100u + dbg_slot], meta, memory_order_relaxed);
                }
            }
            #endif

            // 정석 fix 원칙 (2026-04-23): D4 corner edge geometric reject hack 제거.
            // hack/스케일링/geometric heuristic 금지. Geant4 line-by-line 비교 후
            // 누락된 operation 을 정확히 구현하는 것이 정석. 진행 중.

            #if ENABLE_DIAGNOSTICS
            // === V16-GPU: 광자별 boundary attempts count → photon.flags 상위 5비트 사용 (max 31) ===
            if (!hasSurface && matIdIn != matIdOut) {
                photon.flags = (photon.flags & 0x07FFFFFFu) | (((min((photon.flags >> 27) + 1u, 31u)) & 0x1Fu) << 27);
            }
            // === V15-GPU: Air→Glass entry attempts + reflect 카운트 (CPU와 비교용) ===
            // slot 4000 = Air→Glass attempts
            // slot 4001 = Air→Glass reflect
            if (!hasSurface && matIdIn != matIdOut && bnd_n1 < bnd_n2) {  // Air→Glass entry
                atomic_fetch_add_explicit(&diagCounters[4000], 1u, memory_order_relaxed);
                // reflect 결정 미리 peek
                float3 dbg_dir15 = float3(photon.direction);
                float3 dbg_pol15 = float3(photon.polarization);
                float3 dbg_N15 = closestNormal;
                float dbg_cosI15 = -dot(dbg_dir15, dbg_N15);
                if (dbg_cosI15 < 0.0) { dbg_N15 = -dbg_N15; dbg_cosI15 = -dbg_cosI15; }
                float dbg_Rs15, dbg_Rp15;
                float dbg_R15 = FresnelReflectance(dbg_cosI15, bnd_n1, bnd_n2,
                                                    dbg_dir15, dbg_N15, dbg_pol15,
                                                    dbg_Rs15, dbg_Rp15);
                RandomState rng_peek15 = rng;
                float dbg_rand15 = RandUniform(rng_peek15);
                if (dbg_rand15 < dbg_R15) {
                    atomic_fetch_add_explicit(&diagCounters[4001], 1u, memory_order_relaxed);
                }
            }
            // === 검증 13: 실제 (rand, R) 페어 dump ===
            //   ProcessBoundaryFresnel 호출 직전에 RandomState copy 후 peek해서 next random 값 알아냄.
            //   slot 2500 = atomic counter
            //   slot 2600 + i*4 + (0..3) = (cosI, R, rand_peek, reflect_decision_flag)
            //   non-SameMaterial Glass→Air boundary만 capture.
            if (!hasSurface && matIdIn != matIdOut && bnd_n1 > bnd_n2) {
                float3 dbg_dir3 = float3(photon.direction);
                float3 dbg_pol3 = float3(photon.polarization);
                float3 dbg_N4 = closestNormal;
                float dbg_cosI3 = -dot(dbg_dir3, dbg_N4);
                if (dbg_cosI3 < 0.0) { dbg_N4 = -dbg_N4; dbg_cosI3 = -dbg_cosI3; }
                float dbg_Rs3, dbg_Rp3;
                float dbg_R3 = FresnelReflectance(dbg_cosI3, bnd_n1, bnd_n2,
                                                   dbg_dir3, dbg_N4, dbg_pol3,
                                                   dbg_Rs3, dbg_Rp3);
                // RandomState peek: copy로 RandUniform 호출 → rng 원본 진화 안 함
                RandomState rng_peek = rng;
                float dbg_rand = RandUniform(rng_peek);
                bool dbg_will_reflect = (dbg_rand < dbg_R3);

                uint slot13 = atomic_fetch_add_explicit(&diagCounters[2500], 1u, memory_order_relaxed);
                if (slot13 < 200u) {
                    uint base13 = 2600u + slot13 * 4u;
                    atomic_store_explicit(&diagCounters[base13+0u], as_type<uint>(dbg_cosI3), memory_order_relaxed);
                    atomic_store_explicit(&diagCounters[base13+1u], as_type<uint>(dbg_R3), memory_order_relaxed);
                    atomic_store_explicit(&diagCounters[base13+2u], as_type<uint>(dbg_rand), memory_order_relaxed);
                    atomic_store_explicit(&diagCounters[base13+3u], dbg_will_reflect ? 1u : 0u, memory_order_relaxed);
                }
            }

            // === 검증 11: dielectric boundary R 직접 dump (SameMaterial 제외) ===
            //   slot 1200 = atomic counter
            //   slot 1300 + i*8 + (0..7) = (cosI, n1, n2, R, polX, polY, polZ, Rs)
            //   첫 100개 non-SameMaterial boundary event capture per dispatch.
            if (!hasSurface && matIdIn != matIdOut && bnd_n1 > bnd_n2) {  // Glass→Air only
                float3 dbg_dir2 = float3(photon.direction);
                float3 dbg_pol2 = float3(photon.polarization);
                float3 dbg_N3 = closestNormal;
                float dbg_cosI2 = -dot(dbg_dir2, dbg_N3);
                if (dbg_cosI2 < 0.0) { dbg_N3 = -dbg_N3; dbg_cosI2 = -dbg_cosI2; }
                float dbg_Rs2, dbg_Rp2;
                float dbg_R2 = FresnelReflectance(dbg_cosI2, bnd_n1, bnd_n2,
                                                   dbg_dir2, dbg_N3, dbg_pol2,
                                                   dbg_Rs2, dbg_Rp2);
                uint slot11 = atomic_fetch_add_explicit(&diagCounters[1200], 1u, memory_order_relaxed);
                if (slot11 < 100u) {
                    uint base11 = 1300u + slot11 * 8u;
                    atomic_store_explicit(&diagCounters[base11+0u], as_type<uint>(dbg_cosI2), memory_order_relaxed);
                    atomic_store_explicit(&diagCounters[base11+1u], as_type<uint>(bnd_n1), memory_order_relaxed);
                    atomic_store_explicit(&diagCounters[base11+2u], as_type<uint>(bnd_n2), memory_order_relaxed);
                    atomic_store_explicit(&diagCounters[base11+3u], as_type<uint>(dbg_R2), memory_order_relaxed);
                    atomic_store_explicit(&diagCounters[base11+4u], as_type<uint>(dbg_pol2.x), memory_order_relaxed);
                    atomic_store_explicit(&diagCounters[base11+5u], as_type<uint>(dbg_pol2.y), memory_order_relaxed);
                    atomic_store_explicit(&diagCounters[base11+6u], as_type<uint>(dbg_pol2.z), memory_order_relaxed);
                    atomic_store_explicit(&diagCounters[base11+7u], as_type<uint>(dbg_Rs2), memory_order_relaxed);
                }
            }
            #endif

            if (hasSurface) {
                ProcessBoundaryWithSurface(photon, rng, closestNormal,
                                          surfaces[hitTri.surfaceId],
                                          surfLUTs[hitTri.surfaceId],
                                          matLUTs, matIdIn, matIdOut,
                                          diagCounters);
            } else {
                ProcessBoundaryFresnel(photon, rng, closestNormal,
                                      matIdOut, materials, matIdIn,
                                      diagCounters, bnd_n1, bnd_n2);
            }
            // 정석 fix (2026-05-06, fee99bf 패턴 확장): boundary processing 후
            // photon.direction 변할 수 있음 (reflection/refraction/TIR). 직전
            // mesh BVH hit 의 lastHitTriId 가 stale → 다음 BVH cast 시 합법한
            // surface skip → 광자 lost. Mirror multi-bounce 시나리오에서 50% lost.
            // Rayleigh/Mie/WLS 만 reset 하던 fee99bf fix 를 boundary 도 확장.
            photon.lastHitTriId = -1;

            #if ENABLE_DIAGNOSTICS
            // 진단: 경계면 이벤트 카운터 + 궤적 덤프
            {
                bool transmitted = (photon.materialId != matBefore);
                // Expert E: Glass exit reflected count (face별)
                // EMULATE_G4_TIR_PUSH=1 시 TIR 도 photon.materialId 가 matIdOut 으로 바뀌므로
                // transmitted=true 가 되어 misleading. reflectedCount 증가량으로 판정.
                bool dbg_wasReflected = (photon.reflectedCount > dbg_reflBefore);
                if (dbg_isGlassExit && dbg_wasReflected && dbg_faceSlot >= 50) {
                    atomic_fetch_add_explicit(&diagCounters[dbg_faceSlot + 4], 1, memory_order_relaxed);
                }
                // === 검증 12: Glass bottom reflect 후 새 dir/pos/pol/normal dump ===
                if (dbg_isGlassExit && dbg_wasReflected && dbg_faceSlot == 50) {
                    uint slot12 = atomic_fetch_add_explicit(&diagCounters[1500], 1u, memory_order_relaxed);
                    if (slot12 < 100u) {
                        uint base12 = 1600u + slot12 * 9u;
                        float3 newDir = float3(photon.direction);
                        float3 newPos = float3(photon.position);
                        float3 newPol = float3(photon.polarization);
                        atomic_store_explicit(&diagCounters[base12+0u], as_type<uint>(newDir.x), memory_order_relaxed);
                        atomic_store_explicit(&diagCounters[base12+1u], as_type<uint>(newDir.y), memory_order_relaxed);
                        atomic_store_explicit(&diagCounters[base12+2u], as_type<uint>(newDir.z), memory_order_relaxed);
                        atomic_store_explicit(&diagCounters[base12+3u], as_type<uint>(newPos.x), memory_order_relaxed);
                        atomic_store_explicit(&diagCounters[base12+4u], as_type<uint>(newPos.y), memory_order_relaxed);
                        atomic_store_explicit(&diagCounters[base12+5u], as_type<uint>(newPos.z), memory_order_relaxed);
                        atomic_store_explicit(&diagCounters[base12+6u], as_type<uint>(newPol.x), memory_order_relaxed);
                        atomic_store_explicit(&diagCounters[base12+7u], as_type<uint>(newPol.y), memory_order_relaxed);
                        atomic_store_explicit(&diagCounters[base12+8u], as_type<uint>(newPol.z), memory_order_relaxed);
                    }
                }
                // Expert SideX-TIR z-distribution: SideX TIR 광자 z bin 카운트
                // Bins: z ∈ [-100, 0] → 10개 bin (10mm씩), slot 40-49.
                // bin0=[-100,-90], bin1=[-90,-80], ..., bin9=[-10,0]
                if (dbg_isTIR && dbg_wasReflected) {
                    int dbg_zbin = (int)((dbg_zBefore + 100.0f) / 10.0f);
                    if (dbg_zbin < 0) dbg_zbin = 0;
                    if (dbg_zbin > 9) dbg_zbin = 9;
                    atomic_fetch_add_explicit(&diagCounters[40 + dbg_zbin], 1, memory_order_relaxed);
                }
                // Expert reflect_dir (2026-04-22): Glass bottom (-Z, slot=50) reflect dir 정밀도 측정
                // 이론값: dir = (0.4714, 0, +0.8820)
                // 슬롯 80-87 사용 (32-39는 host에서 다른 진단 read 중, 60은 다른 expert 사용)
                // ※ buffer 크기는 host 측에서 128로 확장하여야 함.
                // slot 80: dir.x scaled sum (x*1e5 + 1e5, int [0, 2e5])
                // slot 81: dir.z scaled sum
                // slot 82: dir.x^2 * 1e6
                // slot 83: dir.z^2 * 1e6
                // slot 84: count
                // slot 85: dir.x min as uint (x*1e5 + 1e5 — atomic_min)
                // slot 86: dir.x max as uint (x*1e5 + 1e5 — atomic_max)
                // slot 87: dir.z min as uint (z*1e5 + 1e5 — atomic_min)
                if (dbg_isGlassExit && dbg_wasReflected && dbg_faceSlot == 50) {
                    float3 rd = float3(photon.direction);
                    uint x_scaled = (uint)round(rd.x * 1.0e5f + 1.0e5f);
                    uint z_scaled = (uint)round(rd.z * 1.0e5f + 1.0e5f);
                    uint xx_scaled = (uint)round(rd.x * rd.x * 1.0e6f);
                    uint zz_scaled = (uint)round(rd.z * rd.z * 1.0e6f);
                    atomic_fetch_add_explicit(&diagCounters[80], x_scaled, memory_order_relaxed);
                    atomic_fetch_add_explicit(&diagCounters[81], z_scaled, memory_order_relaxed);
                    atomic_fetch_add_explicit(&diagCounters[82], xx_scaled, memory_order_relaxed);
                    atomic_fetch_add_explicit(&diagCounters[83], zz_scaled, memory_order_relaxed);
                    atomic_fetch_add_explicit(&diagCounters[84], 1, memory_order_relaxed);
                    atomic_fetch_min_explicit(&diagCounters[85], x_scaled, memory_order_relaxed);
                    atomic_fetch_max_explicit(&diagCounters[86], x_scaled, memory_order_relaxed);
                    atomic_fetch_min_explicit(&diagCounters[87], z_scaled, memory_order_relaxed);
                }
                if (dumpTrajectory) {
                    uint tIdx = atomic_fetch_add_explicit(&trajectoryCount, 1, memory_order_relaxed);
                    if (tIdx < MAX_TRAJECTORY_STEPS) {
                        TrajectoryStep ts;
                        ts.position = packed_float3(newPos);
                        ts.direction = packed_float3(float3(photon.direction));
                        ts.normal = packed_float3(closestNormal);
                        ts.stepDist = stepDist;
                        ts.distToBoundary = distToBoundary;
                        ts.distToAbsorb = distToAbsorb;
                        ts.cosI = traj_cosI;
                        ts.n1 = traj_n1;
                        ts.n2 = traj_n2;
                        ts.fresnelR = traj_R;
                        ts.randVal = 0;
                        ts.processType = 0;
                        ts.matIdIn = matIdIn;
                        ts.matIdOut = matIdOut;
                        ts.outcome = transmitted ? 0 : 1;
                        if (traj_R >= 1.0) ts.outcome = 2;
                        if (photon.status == BOUNDARY_ABS) ts.outcome = 3;
                        ts.triangleId = closestTriIdx;
                        ts.volumeId = hitTri.volumeId;
                        ts.photonIdx = idx;
                        trajectoryBuf[tIdx] = ts;
                    }
                }
                if (hasSurface) {
                    atomic_fetch_add_explicit(&diagCounters[0], 1, memory_order_relaxed);
                    if (transmitted) atomic_fetch_add_explicit(&diagCounters[2], 1, memory_order_relaxed);
                    else atomic_fetch_add_explicit(&diagCounters[1], 1, memory_order_relaxed);
                } else {
                    atomic_fetch_add_explicit(&diagCounters[3], 1, memory_order_relaxed);
                    if (transmitted) atomic_fetch_add_explicit(&diagCounters[5], 1, memory_order_relaxed);
                    else atomic_fetch_add_explicit(&diagCounters[4], 1, memory_order_relaxed);
                }
                if (closestTriIdx == lastTriIdx && !transmitted)
                    atomic_fetch_add_explicit(&diagCounters[6], 1, memory_order_relaxed);
                if (stepDist < 0.001)
                    atomic_fetch_add_explicit(&diagCounters[7], 1, memory_order_relaxed);
            }
            #endif

            #if ENABLE_DIAGNOSTICS
            lastTriIdx = closestTriIdx;  // 현재 삼각형 기억 (진단 전용)
            #endif

            // 물질/볼륨 ID 동기화
            currentMatId = photon.materialId;
            if (photon.materialId == matIdOut) {
                // Fix 70c: volumeId는 방향(dirDotN)으로 결정
                // dirDotN < 0 = 볼륨 진입, dirDotN > 0 = 볼륨 퇴출
                // 기존 물질 비교 방식은 Air/Air 경계(ImagingSensor 등)에서 실패
                float dirDotN_vol = dot(float3(photon.direction), closestNormal);
                if (dirDotN_vol < 0.0) {
                    // 볼륨 진입 → hitTri의 volumeId 사용
                    photon.volumeId = newVolumeId;
                } else {
                    // 볼륨 퇴출 → 부모(월드) 볼륨
                    photon.volumeId = 0;
                }
            }
            photon.flags++;  // 경계 상호작용 카운트

            // Fix (2026-04-22): boundary 진입 시 spurious weight=1.0 hit 등록을
            // 제거. 다음 step 의 step-AABB intersection 검사 (line 805-907) 가
            // 진입 segment 의 path length 를 정확히 누적하므로 redundant 했고,
            // weight=1.0 (mm) 가 fluence scorer 에서 path length 로 해석되어
            // 광자당 ~1mm spurious 누적이 multi-bounce 시 큰 fluence 부풀림으로
            // 이어졌음. CPU TOPAS GenericFluenceScorer 는 step length 만 누적
            // → GPU 도 step-AABB 누적만 사용하여 일치.

            // Fix 34: Coincident surface 연쇄 통과
            // fluence에 영향 없음 확인됨 (Fix 57/61 검증). Fix 63: production에서 비활성화.
            #if ENABLE_DIAGNOSTICS
            if (photon.status == ALIVE) {
                int chainPrevTriIdx = closestTriIdx;
                for (int chainStep = 0; chainStep < 4; chainStep++) {
                    // 현재 위치에서 매우 가까운 거리에 다른 삼각형이 있는지 검색
                    // 2026-05-08 fp32 precision fix: BVH centroid offset 적용
                    float3 bvhCentroidChain = float3(config.bvhCentroidX, config.bvhCentroidY, config.bvhCentroidZ);
                    ray rChain(float3(photon.position) - bvhCentroidChain, float3(photon.direction), 0.0, 2e-4);
                    intersector<triangle_data> interChain;
                    interChain.accept_any_intersection(false);
                    auto resChain = interChain.intersect(rChain, accelStruct);

                    if (resChain.type != intersection_type::triangle) break;

                    int chainTriIdx = (int)resChain.primitive_id;
                    if (chainTriIdx == chainPrevTriIdx) {
                        // 같은 면 → 약간 이동 후 재시도
                        photon.position = packed_float3(
                            float3(photon.position) + 1e-5 * float3(photon.direction));
                        continue;
                    }

                    // 다른 면 발견 → 물질 전환 처리
                    TriangleAttr chainTri = triangles[chainTriIdx];  // Fix 71: 32B
                    float3 chainNormal = float3(chainTri.normal);
                    uint chainMatIn = photon.materialId;
                    float chainDirDotN = dot(float3(photon.direction), chainNormal);
                    uint chainMatOut = chainDirDotN < 0 ?
                        chainTri.materialIdInside : chainTri.materialIdOutside;

                    if (chainMatOut >= MAX_MATERIALS) break;
                    if (chainMatIn == chainMatOut) break;  // 같은 물질 → 불필요

                    // 경계 처리 (Fresnel 또는 Surface)
                    bool chainHasSurf = (chainTri.surfaceId > 0 &&
                                         chainTri.surfaceId < MAX_SURFACES);
                    if (chainHasSurf) {
                        ProcessBoundaryWithSurface(photon, rng, chainNormal,
                            surfaces[chainTri.surfaceId],
                            surfLUTs[chainTri.surfaceId],
                            matLUTs, chainMatIn, chainMatOut,
                            diagCounters);
                    } else {
                        ProcessBoundaryFresnel(photon, rng, chainNormal,
                            chainMatOut, materials, chainMatIn, diagCounters);
                    }

                    currentMatId = photon.materialId;
                    chainPrevTriIdx = chainTriIdx;
                    photon.flags++;

                    if (photon.status != ALIVE) break;
                    break;  // 한 단계만 처리 (보통 1개의 coincident face)
                }
            }
            #endif  // ENABLE_DIAGNOSTICS — coincident chain

            #if ENABLE_DIAGNOSTICS
            // Fix 36 디버그: 경계 충돌 정보 추적 (Fix 78e: _padding 제거로 인코딩 저장소 삭제)
            {
                uint volBits = as_type<uint>(photon.weight);
                volBits |= (1u << (hitTri.volumeId & 31));
                photon.weight = as_type<float>(volBits);
            }
            #endif

        } else if (stepDist == distToWLS) {
            // M7 수정: WLS 흡수 → 재방출 (흡수보다 우선)
            if (mat.wlsComponent.count > 0) {
                // WLS 스펙트럼에서 새 에너지 샘플링
                float totalArea = 0.0;
                for (uint i = 1; i < mat.wlsComponent.count; i++) {
                    totalArea += 0.5 *
                        (mat.wlsComponent.values[i-1] + mat.wlsComponent.values[i]) *
                        (mat.wlsComponent.energies[i] - mat.wlsComponent.energies[i-1]);
                }
                if (totalArea > 0.0) {
                    float target = RandUniform(rng) * totalArea;
                    float cumArea = 0.0;
                    for (uint i = 1; i < mat.wlsComponent.count; i++) {
                        float dArea = 0.5 *
                            (mat.wlsComponent.values[i-1] + mat.wlsComponent.values[i]) *
                            (mat.wlsComponent.energies[i] - mat.wlsComponent.energies[i-1]);
                        if (cumArea + dArea >= target) {
                            // Fix: dArea=0일 때 NaN 방지 (중복 에너지 또는 값=0인 구간)
                            float frac = (dArea > 0.0) ? (target - cumArea) / dArea : 0.5;
                            photon.energy = mix(mat.wlsComponent.energies[i-1],
                                               mat.wlsComponent.energies[i], frac);
                            break;
                        }
                        cumArea += dArea;
                    }
                }
                photon.wavelength = HC_EVNM / photon.energy;

                // 시간 지연
                photon.time += -mat.wlsTimeConstant * MOP_LOG(max(RandUniform(rng), 1e-10));

                // 새로운 등방성 방향
                float ct = 2.0 * RandUniform(rng) - 1.0;
                float st = MOP_SQRT(max(0.0, 1.0 - ct * ct));
                float ph = TWO_PI * RandUniform(rng);
                float3 newDir = float3(st * MOP_COS(ph), st * MOP_SIN(ph), ct);
                photon.direction = packed_float3(newDir);

                // 새 편광 — G4OpWLS.cc:189-195 정석: 운동량 수직면에서 azimuth 균일 랜덤
                // (기존: deterministic cross(newDir,upP) = dir-pol 상관 → 다중반사 Fresnel/TIR s/p 편향)
                {
                    float cosp_w = MOP_COS(ph);
                    float sinp_w = MOP_SIN(ph);
                    // G4 초기 편광 (cost*cosp, cost*sinp, -sint) = (ct*cosp, ct*sinp, -st), newDir에 수직
                    float3 pol0 = float3(ct * cosp_w, ct * sinp_w, -st);
                    float3 perp_w = cross(newDir, pol0);
                    float phi2 = TWO_PI * RandUniform(rng);  // G4 두번째 난수로 azimuth 랜덤화
                    photon.polarization = packed_float3(normalize(MOP_COS(phi2) * pol0 + MOP_SIN(phi2) * perp_w));
                }

                // WLS 재방출: 계속 전파
                photon.lastHitTriId = -1;  // 산란 후 stale dedup reject 방지 (Rayleigh fix 동일)
                continue;
            } else {
                // WLS 스펙트럼 없으면 흡수
                photon.status = ABSORBED;
                break;
            }

        } else if (stepDist == distToRayleigh) {
            // 레일리 산란 (편광 포함)
            DoRayleighScattering(photon, rng);
            // 정석 fix (2026-04-27): Rayleigh/Mie/WLS 후 lastHitTriId reset.
            // Self-intersection dedup (line 1003) 은 boundary 직후 같은 triangle
            // 재방문을 막기 위한 것 — 산란으로 direction 이 바뀌면 다음 BVH hit 의
            // 같은 triangle 은 LEGITIMATE NEW HIT 임. Reset 안 하면 stale lastHitTri
            // 가 영구 reject 되어 distToBoundary 가 distant face 로 점프 → Rayleigh
            // 가 boundary 너머로 photon teleport (U3 halo +78% bias root cause).
            photon.lastHitTriId = -1;

        } else if (stepDist == distToMie) {
            // 미 산란 (편광 포함)
            DoMieScattering(photon, rng,
                mat.miehgForward, mat.miehgBackward, mat.miehgForwardRatio);
            photon.lastHitTriId = -1;  // 산란 후 stale dedup reject 방지 (Rayleigh fix 동일)

        } else if (stepDist == distToAbsorb) {
            // 벌크 흡수 — 검출기 물질이면 DETECTED로 처리
            #if ENABLE_DIAGNOSTICS
            if (dumpTrajectory) {
                uint tIdx = atomic_fetch_add_explicit(&trajectoryCount, 1, memory_order_relaxed);
                if (tIdx < MAX_TRAJECTORY_STEPS) {
                    TrajectoryStep ts = {};
                    ts.position = packed_float3(newPos);
                    ts.stepDist = stepDist;
                    ts.distToAbsorb = distToAbsorb;
                    ts.distToBoundary = distToBoundary;
                    ts.processType = 1;
                    ts.matIdIn = currentMatId;
                    ts.outcome = 4;
                    ts.photonIdx = idx;
                    trajectoryBuf[tIdx] = ts;
                }
            }
            #endif
            if (mat.isDetector) {
                photon.status = DETECTED;
            } else {
                photon.status = ABSORBED;
            }
            #if ENABLE_DIAGNOSTICS
            // [Fix82-DEBUG] count absorbs by matId for SF1 mismatch tracking
            // slot 110 = total bulk absorb in matId=2 (BC408Black = ScRefl)
            // slot 111 = total bulk absorb in matId=1 (Buapfcfm = Sci/Wrap)
            // slot 112 = total bulk absorb in other matIds
            if (currentMatId == 2u) {
                atomic_fetch_add_explicit(&diagCounters[110], 1u, memory_order_relaxed);
            } else if (currentMatId == 1u) {
                atomic_fetch_add_explicit(&diagCounters[111], 1u, memory_order_relaxed);
            } else {
                atomic_fetch_add_explicit(&diagCounters[112], 1u, memory_order_relaxed);
            }
            #endif
            break;
        }

        // Fix 36: 비경계 프로세스에서는 weight/padding 유지 (경계 정보 보존)

        // SF1 (post-boundary refactor 2026-04-24): CPU TsScoreSurfaceTrackCount 등가
        // 5명 합의 알고리즘 (TOPAS 전문가 강조 — materialId 게이트 1차):
        //   1. material 변경 = volume cross 발생 (G4 fGeomBoundary 등가, 핵심 게이트)
        //   2. dPre vs dPost sign-XOR로 어느 surface 통과했는지 매칭
        //   3. cosθ = post-boundary direction · normal로 in/out 판정
        //   4. Wächter-Binder push 적용된 후 최종 photon.position/direction 사용
        //   5. tolerance 1e-7mm (GPU 권장, push 1e-4mm 대비 안전)
        // SF1 deficit fix2 (2026-04-24): materialId gate도 제거.
        // CPU TsScoreSurfaceTrackCount는 fGeomBoundary 기반 (volume 변경)이고,
        // Air→Air 같은 same-material 인접 volume도 fGeomBoundary 발생.
        // GPU shader는 volumeId 추적 안 함 → signed-distance crossing만으로 판정.
        // 반사 광자가 surface 만나서 sign 안 바뀌는 케이스는 자연스럽게 skip됨.
        #if ENABLE_DIAGNOSTICS
        // [Fix82-DEBUG] mat change tracking — count Sci(1)→ScRefl(2) transitions
        // Compare with SF1 R_sf1 fires to detect missed crossings.
        bool dbg_matChange12 = (sf1_stepStartMat == 1u && currentMatId == 2u);
        if (dbg_matChange12) {
            atomic_fetch_add_explicit(&diagCounters[120], 1u, memory_order_relaxed);
            // Also log dPre/dPost of the z=-100 plane crossing for diagnosis
            float dPre_z100  = sf1_stepStartPos.z + 100.0f;
            float dPost_z100 = float3(photon.position).z + 100.0f;
            // slot 121: stepDist == 0 cases
            if (stepDist <= 0.0f) {
                atomic_fetch_add_explicit(&diagCounters[121], 1u, memory_order_relaxed);
            }
            // slot 122: sf1_oldP already at z<-100 (already in ScRefl region)
            if (dPre_z100 < 0) {
                atomic_fetch_add_explicit(&diagCounters[122], 1u, memory_order_relaxed);
            }
            // slot 123: sf1_newP at z>=-100 (didn't cross plane post-step)
            if (dPost_z100 >= 0) {
                atomic_fetch_add_explicit(&diagCounters[123], 1u, memory_order_relaxed);
            }
            // slot 124: both signs same (no sign change)
            if ((dPre_z100 < 0) == (dPost_z100 < 0)) {
                atomic_fetch_add_explicit(&diagCounters[124], 1u, memory_order_relaxed);
            }
        }
        #endif

        if (config.numSurfaces > 0 && stepDist > 0.0f) {
            // SF1 over-count fix (2026-04-25): geometric crossing direction must
            // agree with cosTheta from post-boundary direction.
            // Bug: at iter that processes Fresnel reflect, sf1_oldP can be slightly
            // past plane (fp32 imprecision at boundary stop) → sign-XOR + cosTheta
            // (post-direction = reflected dir) wrongly fires going_in for incident
            // photon's reflect step. Fix: skip fire if geometric direction implied
            // by sign change disagrees with cosTheta direction (= boundary processing
            // changed photon direction mid-iter, sign-XOR is artifact).
            float3 sf1_oldP = sf1_stepStartPos;
            float3 sf1_newP = float3(photon.position);  // post-boundary, post-WB-push
            float3 sf1_postDir = float3(photon.direction);  // post-boundary direction
            uint nSurf2 = config.numSurfaces;  // device buffer — host 가 크기 보장
            for (uint si2 = 0; si2 < nSurf2; si2++) {
                SurfaceDef surf = surfaceDefs[si2];
                if (surf.enabled == 0) continue;

                float3 origin = float3(surf.origin);
                float3 normal = float3(surf.normal);
                float dPre  = dot(sf1_oldP - origin, normal);
                float dPost = dot(sf1_newP - origin, normal);

                // sign-XOR (GPU 권장: underflow 안전)
                bool signDiff = (dPre < 0.0f) ^ (dPost < 0.0f);
                if (!signDiff) continue;

                // tolerance 1e-7mm: 두 부호 모두 명확해야 함
                const float kPlanePostTol = 1e-7f;
                if (abs(dPre) < kPlanePostTol && abs(dPost) < kPlanePostTol) continue;

                // Bounded plane check: crossing point가 surface 내부에 있어야 함
                float t = dPre / (dPre - dPost);  // 항상 [0,1] (signDiff true 보장)
                float3 crossP = sf1_oldP + t * (sf1_newP - sf1_oldP);
                float3 toCross = crossP - origin;
                float u = dot(toCross, float3(surf.axisU));
                float v = dot(toCross, float3(surf.axisV));
                if (abs(u) > surf.halfExtentU || abs(v) > surf.halfExtentV) continue;

                // post-boundary direction · normal로 in/out 판정 (CPU 의미론)
                float cosTheta = dot(sf1_postDir, normal);

                // SF1 over-count fix: geometric sign change direction must match cosTheta sign.
                // signDiff=true means exactly one of (dPre<0) and (dPost<0) is true.
                // Going in (-normal): dPre>=0, dPost<0 → cosTheta should be < 0.
                // Going out (+normal): dPre<0, dPost>=0 → cosTheta should be > 0.
                // Mismatch = boundary processing reversed direction mid-iter (spurious fire).
                bool geomGoingIn = (dPre >= 0.0f) && (dPost < 0.0f);
                bool sf1GoingIn  = (cosTheta < 0.0f);
                if (geomGoingIn != sf1GoingIn) continue;

                uint sIdx = atomic_fetch_add_explicit(&surfaceHitCount, 1u, memory_order_relaxed);
                if (sIdx < maxSurfaceHits) {
                    SurfaceHit sh;
                    sh.position = packed_float3(crossP);
                    sh.direction = packed_float3(sf1_postDir);
                    sh.energy = photon.energy;
                    sh.cosTheta = cosTheta;
                    sh.photonWeight = photon.weight;
                    sh.time = photon.time;
                    sh.surfaceId = surf.surfaceId;
                    sh.flags = (cosTheta < 0.0f) ? 1u : 0u;  // bit 0: 1=going_in
                    sh.trackId = idx;
                    sh.materialIdFrom = sf1_stepStartMat;
                    sh.materialIdTo = currentMatId;
                    sh._pad = 0u;
                    surfaceHits[sIdx] = sh;
                }
            }
        }

        // World boundary clamp 결과 (parallel-world scoring 정석 fix):
        // transit-hit 정상 기록 후 step 종료 시점에 OUT_OF_WORLD 표시.
        if (willExitWorld) {
            photon.status = OUT_OF_WORLD;
            break;
        }

        // 최대 스텝 체크
        if (photon.stepCount >= config.maxStepsPerPhoton) {
            photon.status = MAX_STEPS;
            break;
        }
    }

    #if ENABLE_DIAGNOSTICS
    // 진단: 광자별 최종 통계 기록
    atomic_fetch_add_explicit(&diagCounters[8], photon.stepCount, memory_order_relaxed);
    if (photon.status < 7)
        atomic_fetch_add_explicit(&diagCounters[9 + photon.status], 1, memory_order_relaxed);

    // [FateDiag 2026-04-27] OUT_OF_WORLD photon 의 final position 으로 face 분류
    // BC408 = ±25 (X,Y) × ±150 (Z), World = ±200 (X,Y) × ±400 (Z)
    // (slot 130-139 free)
    //   slot 130: -X face escape (x <= -25 mm boundary)
    //   slot 131: +X face escape
    //   slot 132: -Y face escape
    //   slot 133: +Y face escape
    //   slot 134: -Z face escape (backward end, z <= -150)
    //   slot 135: +Z face escape (forward end, z >= +150)
    //   slot 136: ABSORBED inside BC408 (currentMatId Buapfcfm 등)
    //   slot 137: ABSORBED in air or other
    if (photon.status == OUT_OF_WORLD) {
        float3 fp = float3(photon.position);
        float ax = fabs(fp.x), ay = fabs(fp.y), az = fabs(fp.z);
        // dominant face = largest absolute component projected past BC408 face
        float dx = ax - 25.0f, dy = ay - 25.0f, dz = az - 150.0f;
        if (dx >= dy && dx >= dz) {
            atomic_fetch_add_explicit(&diagCounters[fp.x < 0 ? 130 : 131], 1u, memory_order_relaxed);
        } else if (dy >= dz) {
            atomic_fetch_add_explicit(&diagCounters[fp.y < 0 ? 132 : 133], 1u, memory_order_relaxed);
        } else {
            atomic_fetch_add_explicit(&diagCounters[fp.z < 0 ? 134 : 135], 1u, memory_order_relaxed);
        }
    } else if (photon.status == ABSORBED) {
        // currentMatId 1=Buapfcfm (BC408 inside)
        if (currentMatId == 1u)
            atomic_fetch_add_explicit(&diagCounters[136], 1u, memory_order_relaxed);
        else
            atomic_fetch_add_explicit(&diagCounters[137], 1u, memory_order_relaxed);
    }
    #endif

    // M8 수정: 히트 기록 시 버퍼 경계 체크
    if (photon.status == DETECTED) {
        uint hitIdx = atomic_fetch_add_explicit(&hitCount, 1, memory_order_relaxed);
        if (hitIdx < maxHits) {
            Hit hit;
            hit.position = photon.position;
            hit.time = photon.time;
            hit.energy = photon.energy;
            hit.wavelength = photon.wavelength;
            hit.volumeId = photon.volumeId;
            hit.parentTrackId = photonMeta[idx].parentTrackId;  // Fix 78측정-6: cold meta
            hit.flags = photon.flags;
            hits[hitIdx] = hit;
        }
    }

    // 결과 기록
    photons[idx] = photon;
}
