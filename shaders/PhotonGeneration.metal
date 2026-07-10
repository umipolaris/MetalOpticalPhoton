/**
 * PhotonGeneration.metal
 * 신틸레이션 및 체렌코프 광자 생성 커널
 *
 * TOPAS 파라미터 매핑:
 *   - u:Ma/XXX/ScintillationYield → 에너지 침적 기반 광자 수
 *   - u:Ma/XXX/YieldRatio → fast/slow 성분 비율
 *   - d:Ma/XXX/FastTimeConstant, SlowTimeConstant → 시간 분포
 *   - uv:Ma/XXX/FastComponent, SlowComponent → 에너지 스펙트럼
 */

#include <metal_stdlib>
using namespace metal;

#include "Common.h"

// ============================================================
// 스펙트럼 CDF 샘플링 (m5 수정: 정확한 역 CDF)
// ============================================================
inline float SampleSpectrum(device const PropertyTable& spectrum, thread RandomState& rng) {
    if (spectrum.count <= 1) {
        return spectrum.count == 1 ? spectrum.energies[0] : PHOTON_ENERGY_MIN;
    }

    // CDF 누적 면적 계산
    float totalArea = 0.0;
    for (uint i = 1; i < spectrum.count; i++) {
        float dE = spectrum.energies[i] - spectrum.energies[i-1];
        if (dE > 0.0) {
            totalArea += 0.5 * (spectrum.values[i-1] + spectrum.values[i]) * dE;
        }
    }

    if (totalArea <= 0.0) return spectrum.energies[0];

    float target = RandUniform(rng) * totalArea;
    float cumArea = 0.0;

    for (uint i = 1; i < spectrum.count; i++) {
        float dE = spectrum.energies[i] - spectrum.energies[i-1];
        if (dE <= 0.0) continue;

        float v0 = spectrum.values[i-1];
        float v1 = spectrum.values[i];
        float dArea = 0.5 * (v0 + v1) * dE;

        if (cumArea + dArea >= target) {
            // 사다리꼴 내에서 정확한 위치 계산
            float remaining = target - cumArea;
            // 선형 보간 PDF 내 역 CDF: v0 + (v1-v0)*t/dE 적분 = t*(v0 + (v1-v0)*t/(2*dE))
            // 간단화: 선형 보간으로 근사
            // Fix: dArea=0일 때 (스펙트럼 값이 0인 구간) NaN 방지
            float frac = (dArea > 0.0) ? remaining / dArea : 0.5;
            return mix(spectrum.energies[i-1], spectrum.energies[i], frac);
        }
        cumArea += dArea;
    }

    return spectrum.energies[spectrum.count - 1];
}

// ============================================================
// 신틸레이션 광자 생성
// ============================================================
kernel void generateScintillationPhotons(
    device const Genstep*      gensteps    [[buffer(0)]],
    device const MaterialGPU*  materials   [[buffer(1)]],
    device Photon*             photons     [[buffer(2)]],
    constant SimConfig&        config      [[buffer(3)]],
    device atomic_uint&        photonCount [[buffer(4)]],
    device const uint*         genstepOffsets [[buffer(5)]],
    device PhotonMeta*         photonMeta  [[buffer(6)]],  // Fix 78측정-6: cold meta
    device atomic_uint*        diagCounters [[buffer(7)]],  // [EmitDiag] z bin counter
    uint gid [[thread_position_in_grid]])
{
    // 어떤 genstep의 몇 번째 광자인지 결정
    uint gsIdx = 0;
    uint localPhotonIdx = gid;

    // Fix: config.numGensteps를 루프 상한으로 사용 (OOB 방지)
    uint numGS = config.numGensteps;
    while (gsIdx < numGS &&
           gid >= genstepOffsets[gsIdx + 1]) {
        gsIdx++;
    }
    if (gsIdx >= numGS) return;  // 범위 초과 안전 체크
    localPhotonIdx = gid - genstepOffsets[gsIdx];

    Genstep gs = gensteps[gsIdx];
    if (gs.genType != 0) return;  // SCINTILLATION만 처리

    device const MaterialGPU& mat = materials[gs.materialId];

    // RanecuEngine 시드 초기화 (광자별 독립, kernelSalt=0: scintillation)
    // 2026-05-12 fix: gid + globalPhotonOffset 로 batch 마다 unique RNG (RNG repetition bug fix)
    RandomState rng = InitRandomState(gid + config.globalPhotonOffset, config.randomSeed, 0u);

    // m1 수정: genstep에서 yieldRatio 사용 (물질별 정확한 값)
    float yieldRatio = gs.yieldRatio;
    bool isFast = RandUniform(rng) < yieldRatio;

    // 에너지 스펙트럼에서 샘플링
    device const PropertyTable* spectrum = isFast ? &mat.fastComponent : &mat.slowComponent;
    float energy = SampleSpectrum(*spectrum, rng);

    // 시간 분포: 지수 감쇠
    float timeConstant = isFast ? mat.fastTimeConstant : mat.slowTimeConstant;
    float dt = -timeConstant * MOP_LOG(max(RandUniform(rng), 1e-10));

    // 등방 방향 샘플링
    float3 dir;
    {
        float cosTheta = 2.0f * RandUniform(rng) - 1.0f;
        float sinTheta = MOP_SQRT(max(0.0f, 1.0f - cosTheta * cosTheta));
        float phi = TWO_PI * RandUniform(rng);
        dir = float3(sinTheta * MOP_COS(phi), sinTheta * MOP_SIN(phi), cosTheta);
    }
    dir = normalize(dir);

    // 편광: 신틸레이션은 무편광 (랜덤 편광 방향)
    float3 perpDir;
    if (abs(dir.x) < 0.9)
        perpDir = normalize(cross(dir, float3(1, 0, 0)));
    else
        perpDir = normalize(cross(dir, float3(0, 1, 0)));

    float polAngle = TWO_PI * RandUniform(rng);
    float3 pol = normalize(MOP_COS(polAngle) * perpDir +
                           MOP_SIN(polAngle) * cross(dir, perpDir));

    // Fix 53: Geant4 호환 — 광자 위치를 step을 따라 랜덤 배치
    float posRand = RandUniform(rng);
    float3 genPos = float3(gs.position) + posRand * gs.stepLength * float3(gs.direction);

    // 광자 기록
    Photon p;
    p.position = packed_float3(genPos);
    p.direction = packed_float3(dir);
    p.polarization = packed_float3(pol);
    p.energy = energy;
    p.wavelength = HC_EVNM / energy;
    p.time = gs.time + dt;
    p.weight = 1.0;
    p.status = ALIVE;

    p.volumeId = 0;
    p.materialId = gs.materialId;
    p.stepCount = 0;
    p.flags = 0;
    p.reflectedCount = 0;
    p.lastHitTriId = -1;  // Chroma 정석 fix init
    photons[gid] = p;
    // Fix 78측정-6: meta 별도 write
    PhotonMeta m;
    m.genstepId = gsIdx;
    m.parentTrackId = gs.parentTrackId;
    photonMeta[gid] = m;

    // [EmitDiag 2026-04-27] emit z bin histogram (300 bin × 1mm, range -150~+150)
    // slot 1300 + bin (free range)
    if (diagCounters) {
        int z_bin = (int)floor(genPos.z + 150.0f);
        if (z_bin < 0) z_bin = 0;
        if (z_bin >= 300) z_bin = 299;
        atomic_fetch_add_explicit(&diagCounters[1300u + (uint)z_bin], 1u, memory_order_relaxed);

        // [EmitDirDiag 2026-04-28] emit cosθ histogram (20 bins, cosθ ∈ [-1,+1])
        // 4π isotropic 검증용. slot 5000+bin (free range, dump 영역 밖)
        int cz_bin = (int)floor((dir.z + 1.0f) * 10.0f);
        if (cz_bin < 0) cz_bin = 0;
        if (cz_bin >= 20) cz_bin = 19;
        atomic_fetch_add_explicit(&diagCounters[5000u + (uint)cz_bin], 1u, memory_order_relaxed);

        // [EmitDirDiag] emit phi histogram (18 bins, φ ∈ [-π,+π])
        // slot 5020+bin
        float phi_emit = atan2(dir.y, dir.x);  // [-π, +π]
        int phi_bin = (int)floor((phi_emit + 3.14159265f) / 6.28318530f * 18.0f);
        if (phi_bin < 0) phi_bin = 0;
        if (phi_bin >= 18) phi_bin = 17;
        atomic_fetch_add_explicit(&diagCounters[5020u + (uint)phi_bin], 1u, memory_order_relaxed);
    }
}

// ============================================================
// 체렌코프 광자 생성
// ============================================================
kernel void generateCerenkovPhotons(
    device const Genstep*      gensteps    [[buffer(0)]],
    device const MaterialGPU*  materials   [[buffer(1)]],
    device Photon*             photons     [[buffer(2)]],
    constant SimConfig&        config      [[buffer(3)]],
    device atomic_uint&        photonCount [[buffer(4)]],
    device const uint*         genstepOffsets [[buffer(5)]],
    device PhotonMeta*         photonMeta  [[buffer(6)]],  // Fix 78측정-6
    device atomic_uint*        diagCounters [[buffer(7)]],  // [EmitDiag] (Cerenkov 측정 안 함)
    uint gid [[thread_position_in_grid]])
{
    uint gsIdx = 0;
    uint localPhotonIdx = gid;

    // Fix: config.numGensteps를 루프 상한으로 사용 (OOB 방지)
    uint numGS = config.numGensteps;
    while (gsIdx < numGS &&
           gid >= genstepOffsets[gsIdx + 1]) {
        gsIdx++;
    }
    if (gsIdx >= numGS) return;  // 범위 초과 안전 체크
    localPhotonIdx = gid - genstepOffsets[gsIdx];

    Genstep gs = gensteps[gsIdx];
    if (gs.genType != 1) return;  // CERENKOV만 처리

    device const MaterialGPU& mat = materials[gs.materialId];

    // RanecuEngine 시드 초기화 (kernelSalt=2: cerenkov)
    // 2026-05-12 fix: gid + globalPhotonOffset 로 batch 마다 unique RNG
    RandomState rng = InitRandomState(gid + config.globalPhotonOffset, config.randomSeed, 2u);

    // M3 수정: 체렌코프 에너지 스펙트럼은 1/λ² ∝ E² 분포
    // Frank-Tamm: dN/dE ∝ sin²θ(E) ≈ (1 - 1/(β²n²(E)))
    // 정석 (2026-05-17 fix): G4Cerenkov 와 동일하게 MaxSin2 로 normalize 한
    // rejection. (이전 raw sin² rejection + maxTries=100 cutoff 는 low-β step
    // 에서 silent photon drop → downstream fluence deficit 의 source.)
    // gs.maxCos = betaInverse / maxRI 는 host 가 전달.
    float maxSin2 = max(1.0e-30f, 1.0f - gs.maxCos * gs.maxCos);
    float energy;
    float rindex;
    int maxTries = 10000;  // safety upper bound (사실상 G4 무한 loop)
    do {
        energy = mix(gs.pmin, gs.pmax, RandUniform(rng));
        rindex = InterpolateProperty(mat.refractiveIndex, energy);
        if (rindex <= 0.0) rindex = 1.0;

        // sin²θ = (1-cos)(1+cos) — G4 와 동일한 numerically stable form
        float cosTheta_c = gs.betaInverse / rindex;  // C4 수정: betaInverse/n
        float sin2Theta = (1.0f - cosTheta_c) * (1.0f + cosTheta_c);

        // G4 정석: while (rand * MaxSin2 > sin2Theta) — i.e. accept if rand*MaxSin2 < sin2Theta
        if (sin2Theta > 0.0f && RandUniform(rng) * maxSin2 < sin2Theta) break;

        maxTries--;
    } while (maxTries > 0);

    if (maxTries <= 0) return;

    // C4 수정: cos(θ) = 1/(βn) = betaInverse / n (NOT 1/(betaInverse*n))
    float cosTheta = gs.betaInverse / rindex;
    if (cosTheta > 1.0 || cosTheta < -1.0) return;

    float sinTheta = MOP_SQRT(max(0.0, 1.0 - cosTheta * cosTheta));
    float phi = TWO_PI * RandUniform(rng);

    // 입자 방향 기준 로컬 좌표계
    // Fix: gs.direction이 zero/NaN인 경우 방어적 체크
    float3 rawDir = float3(gs.direction);
    float dirLen = length(rawDir);
    if (dirLen < 1e-6) return;
    float3 particleDir = rawDir / dirLen;

    // 2026-05-18 정석 fix: G4 의 rotateUz(p0) 와 정확 일치 transform 사용.
    // 이전 우리 basis (u = cross(up, p0)/.., v = cross(p0, u), w = p0) 는
    // mathematically equivalent (φ uniform) 하지만 fp32 정밀도 + RNG sequence
    // advance 의 cumulative 결과 광자 fluence mean z 가 -3 mm backward 로 shift
    // (C12 200 MeV/u δ-ray cherenkov 의 measured G/C 0.97).
    // G4 transform (math equivalent but specific basis):
    //   e_x' = (u1*u3/up, u2*u3/up, -up)
    //   e_y' = (-u2/up, u1/up, 0)
    //   e_z' = (u1, u2, u3) = p0
    // particle frame photon momentum: (sinθ·cosφ, sinθ·sinφ, cosθ)
    // global frame: result of rotateUz(p0).
    float cosPhi = MOP_COS(phi);
    float sinPhi = MOP_SIN(phi);
    float u1 = particleDir.x;
    float u2 = particleDir.y;
    float u3 = particleDir.z;
    float up_sq = u1*u1 + u2*u2;

    float3 dir;
    float3 pol;
    if (up_sq > 1e-20f) {
        float up = sqrt(up_sq);
        // photon direction in particle frame
        float px = sinTheta * cosPhi;
        float py = sinTheta * sinPhi;
        float pz = cosTheta;
        dir = float3(
            (u1*u3*px - u2*py)/up + u1*pz,
            (u2*u3*px + u1*py)/up + u2*pz,
            -up*px + u3*pz
        );
        dir = normalize(dir);
        // polarization in particle frame: (cosθ·cosφ, cosθ·sinφ, -sinθ)
        float sx = cosTheta * cosPhi;
        float sy = cosTheta * sinPhi;
        float sz = -sinTheta;
        pol = float3(
            (u1*u3*sx - u2*sy)/up + u1*sz,
            (u2*u3*sx + u1*sy)/up + u2*sz,
            -up*sx + u3*sz
        );
        pol = normalize(pol);
    } else {
        // p0 ~ (0, 0, ±1) — identity or z-flip
        float zsign = (u3 >= 0) ? 1.0f : -1.0f;
        dir = float3(sinTheta * cosPhi, sinTheta * sinPhi, zsign * cosTheta);
        pol = float3(cosTheta * cosPhi, cosTheta * sinPhi, -zsign * sinTheta);
    }

    // 생성 시간: 스텝 내에서 균일 분포 (입자 속도 = c/betaInverse = c*β)
    float particleSpeed = SPEED_OF_LIGHT / gs.betaInverse;
    float dt = RandUniform(rng) * gs.stepLength / particleSpeed;

    Photon p;
    p.position = packed_float3(float3(gs.position) + dt * particleSpeed * particleDir);
    p.direction = packed_float3(dir);
    p.polarization = packed_float3(pol);
    p.energy = energy;
    p.wavelength = HC_EVNM / energy;
    p.time = gs.time + dt;
    p.weight = 1.0;
    p.status = ALIVE;

    p.volumeId = 0;
    p.materialId = gs.materialId;
    p.stepCount = 0;
    p.flags = 0;
    p.reflectedCount = 0;
    p.lastHitTriId = -1;  // Chroma 정석 fix init
    photons[gid] = p;
    // Fix 78측정-6: meta 별도 write
    PhotonMeta m;
    m.genstepId = gsIdx;
    m.parentTrackId = gs.parentTrackId;
    photonMeta[gid] = m;
}


// ============================================================
// 2026-05-12: GPU beam photon generation
// TOPAS Beam source 우회 — 1 genstep → numPhotons photons GPU dispatch.
// Genstep field 재사용 (struct 변경 없음):
//   position    = beam center (world mm)
//   direction   = beam unit direction (must be normalized)
//   charge      = beam energy (eV)        — repurposed
//   numPhotons  = photons to generate (= NumberOfHistoriesInRun)
//   betaInverse = posCutoffX (rect half-width, mm)  — repurposed
//   pmin        = posCutoffY (mm)                   — repurposed
//   pmax        = angSpreadX (Gaussian sigma, rad)  — repurposed
//   maxCos      = angSpreadY (Gaussian sigma, rad)  — repurposed
//   yieldRatio  = angCutoffX (max half-angle, rad)  — repurposed
//   edep        = angCutoffY (rad)                  — repurposed
// 1 thread = 1 photon. global thread id = gid.
// ============================================================
kernel void generateBeamPhotons(
    device const Genstep*      gensteps    [[buffer(0)]],
    device const MaterialGPU*  materials   [[buffer(1)]],
    device Photon*             photons     [[buffer(2)]],
    constant SimConfig&        config      [[buffer(3)]],
    device atomic_uint&        photonCount [[buffer(4)]],
    device const uint*         genstepOffsets [[buffer(5)]],
    device PhotonMeta*         photonMeta  [[buffer(6)]],
    // 2026-05-17 BEAM material auto-detect — photon 위치 가 cyl/box/sphere 안 인지 test
    device const CylinderGeometry* cylinderGeometries [[buffer(7)]],
    device const uint&         numCylGeoms [[buffer(8)]],
    device const BoxGeometry*  boxGeometries [[buffer(9)]],
    device const uint&         numBoxGeoms [[buffer(10)]],
    device const SphereGeometry* sphereGeometries [[buffer(11)]],
    device const uint&         numSphereGeoms [[buffer(12)]],
    uint gid [[thread_position_in_grid]])
{
    // 단일 BEAM genstep 가정 (gsIdx=0). multi-beam 은 추후.
    uint gsIdx = 0;
    if (gid >= gensteps[0].numPhotons) return;

    Genstep gs = gensteps[gsIdx];
    if (gs.genType != 3) return;  // MOP_GEN_BEAM 만 처리

    // RNG: gid + globalPhotonOffset 으로 unique per-photon
    RandomState rng = InitRandomState(gid + config.globalPhotonOffset, config.randomSeed, 5u);

    // beam local basis: u, v perpendicular to direction
    float3 dir0 = float3(gs.direction);
    float3 up = abs(dir0.y) < 0.999 ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 u_axis = normalize(cross(up, dir0));
    float3 v_axis = cross(dir0, u_axis);

    // === Position sampling (TOPAS Beam 정합) ===
    // SimConfig.beamPosDist:  0=None (point), 1=Flat (uniform within cutoff),
    //                         2=Gaussian (σ + cutoff clip).
    // SimConfig.beamPosShape: 0=Rectangle, 1=Ellipse
    float posCutoffX = gs.betaInverse;
    float posCutoffY = gs.pmin;
    float posSigmaX  = config.beamPosSigmaX;
    float posSigmaY  = config.beamPosSigmaY;
    uint  posShape   = config.beamPosShape;  // 0=Rect, 1=Ellipse
    uint  posDist    = config.beamPosDist;   // 0=None, 1=Flat, 2=Gaussian
    float ofsU = 0.0f, ofsV = 0.0f;
    if (posDist != 0u) {
        // Rejection sampling (Flat 또는 Gaussian × shape clip). max 32 tries.
        for (int ptry = 0; ptry < 32; ++ptry) {
            float u1 = RandUniform(rng);
            float u2 = RandUniform(rng);
            float candU, candV;
            if (posDist == 2u) {
                // Box-Muller Gaussian (per-axis σ)
                float r1 = max(u1, 1e-10f);
                float mag = MOP_SQRT(-2.0f * MOP_LOG(r1));
                candU = posSigmaX * mag * MOP_COS(TWO_PI * u2);
                candV = posSigmaY * mag * MOP_SIN(TWO_PI * u2);
            } else {
                // Flat: uniform within cutoff
                candU = (2.0f * u1 - 1.0f) * posCutoffX;
                candV = (2.0f * u2 - 1.0f) * posCutoffY;
            }
            // Shape cutoff
            bool accept;
            if (posShape == 1u) {
                // Ellipse: (u/cutX)² + (v/cutY)² ≤ 1
                float ex = (posCutoffX > 0.0f) ? candU / posCutoffX : 0.0f;
                float ey = (posCutoffY > 0.0f) ? candV / posCutoffY : 0.0f;
                accept = (ex*ex + ey*ey) <= 1.0f;
            } else {
                // Rectangle
                accept = (abs(candU) <= posCutoffX) && (abs(candV) <= posCutoffY);
            }
            if (accept) {
                ofsU = candU; ofsV = candV;
                break;
            }
        }
    }
    float3 genPos = float3(gs.position) + ofsU * u_axis + ofsV * v_axis;

    // === Angular sampling (Marsaglia 1972, TOPAS TsGeneratorBeam 정합) ===
    // SimConfig.beamAngDist: 0=None (dir0), 1=Flat (Marsaglia uniform), 2=Gaussian.
    // TOPAS AngleToMarsagliaCoordinate(θ):
    //   θ ≤ 90°: sin(θ/2)
    //   θ > 90°: cos(θ/2)    (TsGeneratorBeam.cc:297-305 의 부호 분기)
    float sigmaAngX = gs.pmax;       // BeamAngSigmaX rad
    float sigmaAngY = gs.maxCos;     // BeamAngSigmaY rad
    float cutAngX   = gs.yieldRatio; // BeamAngCutoffX rad
    float cutAngY   = gs.edep;       // BeamAngCutoffY rad

    // TOPAS 의 두-분기 Marsaglia 좌표 변환 (TsGeneratorBeam.cc:297-305).
    //   θ ≤ 90°: sin(θ/2),  θ > 90°: cos(θ/2)
    const float HALF_PI = 1.5707963267948966f;
    float spreadMX = (sigmaAngX <= HALF_PI) ? MOP_SIN(sigmaAngX * 0.5f) : MOP_COS(sigmaAngX * 0.5f);
    float spreadMY = (sigmaAngY <= HALF_PI) ? MOP_SIN(sigmaAngY * 0.5f) : MOP_COS(sigmaAngY * 0.5f);
    float cutMX    = (cutAngX   <= HALF_PI) ? MOP_SIN(cutAngX   * 0.5f) : MOP_COS(cutAngX   * 0.5f);
    float cutMY    = (cutAngY   <= HALF_PI) ? MOP_SIN(cutAngY   * 0.5f) : MOP_COS(cutAngY   * 0.5f);

    uint angDist = config.beamAngDist;  // 0=None, 1=Flat, 2=Gaussian, 3=Isotropic(4π)
    float mX = 0.0f, mY = 0.0f;
    float3 newDir = dir0;
    if (angDist == 3u) {
        // 4π isotropic 방출 (dir0 무시). 신틸레이션/WLS 등 등방 소스 재현용.
        // generateScintillationPhotons 의 등방 샘플링과 동일한 수식.
        float cosTheta = 2.0f * RandUniform(rng) - 1.0f;
        float sinTheta = MOP_SQRT(max(0.0f, 1.0f - cosTheta * cosTheta));
        float phi = TWO_PI * RandUniform(rng);
        newDir = float3(sinTheta * MOP_COS(phi), sinTheta * MOP_SIN(phi), cosTheta);
    } else if (angDist != 0u) {
        for (int tries = 0; tries < 32; tries++) {
            float r1 = max(RandUniform(rng), 1e-10f);
            float r2 = RandUniform(rng);
            float candX, candY;
            if (angDist == 2u) {
                // Box-Muller Gaussian × Marsaglia spread (TsGeneratorBeam.cc:270-271)
                float mag = MOP_SQRT(-2.0f * MOP_LOG(r1));
                candX = spreadMX * mag * MOP_COS(TWO_PI * r2);
                candY = spreadMY * mag * MOP_SIN(TWO_PI * r2);
            } else {
                // Flat: uniform within Marsaglia cutoff (TsGeneratorBeam.cc:266-267)
                candX = (2.0f * r1 - 1.0f) * cutMX;
                candY = (2.0f * r2 - 1.0f) * cutMY;
            }
            // Elliptical cutoff (TsGeneratorBeam.cc:277)
            float exx = (cutMX > 0.0f) ? candX / cutMX : 0.0f;
            float eyy = (cutMY > 0.0f) ? candY / cutMY : 0.0f;
            if (exx*exx + eyy*eyy <= 1.0f) {
                mX = candX; mY = candY;
                break;
            }
        }
        // TOPAS local direction (TsGeneratorBeam.cc:288-293)
        float rsq = mX*mX + mY*mY;
        float tmp = MOP_SQRT(max(0.0f, 1.0f - rsq));
        float ldX = 2.0f * mX * tmp;
        float ldY = 2.0f * mY * tmp;
        float ldZ = 1.0f - 2.0f * rsq;
        newDir = ldZ * dir0 + ldX * u_axis + ldY * v_axis;
    }

    // === Polarization (TOPAS 정합) ===
    // SimConfig.beamPolMode: 0=random transverse, 1=fixed (beamPolX/Y/Z 사용).
    // fixed 모드: 사용자 polarization 을 newDir 에 수직 성분만 투영 (perpendicular
    // projection). beamGen 의 newDir 는 angSpread 적용 후 미세 회전돼 있을 수 있어,
    // dir 와 평행 성분 제거가 안전. TOPAS TsVGenerator 는 BeamPos 좌표계의 pol vector
    // 를 그대로 사용하며 dir 회전과 함께 회전 → angSpread 없는 경우 정합 (perpendicular
    // 보정 시 zero 회전 시 결과 동일). angSpread 있는 경우는 미세 차이.
    float3 pol;
    if (config.beamPolMode == 1u) {
        float3 polUser = float3(config.beamPolX, config.beamPolY, config.beamPolZ);
        // Project out parallel component to newDir
        float pDotD = dot(polUser, newDir);
        float3 polPerp = polUser - pDotD * newDir;
        float pmag = length(polPerp);
        if (pmag > 1e-6f) {
            pol = polPerp / pmag;
        } else {
            // Degenerate (pol ∥ dir) — fallback random transverse
            float3 helper = abs(newDir.y) < 0.999 ? float3(0, 1, 0) : float3(1, 0, 0);
            float3 polA = normalize(cross(helper, newDir));
            float3 polB = cross(newDir, polA);
            float polPhi = TWO_PI * RandUniform(rng);
            pol = MOP_COS(polPhi) * polA + MOP_SIN(polPhi) * polB;
        }
    } else {
        // Random transverse (default)
        float3 polPerp = abs(newDir.y) < 0.999 ? float3(0, 1, 0) : float3(1, 0, 0);
        float3 polA = normalize(cross(polPerp, newDir));
        float3 polB = cross(newDir, polA);
        float polPhi = TWO_PI * RandUniform(rng);
        pol = MOP_COS(polPhi) * polA + MOP_SIN(polPhi) * polB;
    }

    // === Energy sampling (TOPAS TsVGenerator.cc:253-280 정합) ===
    // 우선순위: spectrum > mono+Gaussian spread > mono only.
    // 2026-05-19 fix: piecewise linear inverse CDF — 이전 단순 cumW 선형 interp 가
    // j=0 case 의 e_prev=e_curr fallback 으로 50% 광자 가 energies[0] 에 stuck 했음.
    float photonEv = gs.charge;  // nominal mono energy (eV)
    if (config.beamSpectrumNumBins > 0u) {
        float r = RandUniform(rng);
        uint nBins = config.beamSpectrumNumBins;
        if (config.beamSpectrumType == 0u) {
            // Discrete: r cumW 의 first cumW[j] >= r 의 bin
            uint j = nBins - 1u;
            for (uint k = 0u; k < nBins; k++) {
                if (r <= config.beamSpectrumCumWeights[k]) { j = k; break; }
            }
            photonEv = config.beamSpectrumEnergies[j];
        } else {
            // Continuous: piecewise linear PDF 의 inverse CDF (trapezoidal cumW).
            // cumW[0] = 0, cumW[i] = ∫_{E[0]}^{E[i]} w dE / total.
            // r ∈ (cumW[j-1], cumW[j]] → bin j-1 안 의 quadratic root.
            // Bin j-1 안: w(E) = w[j-1] + slope * (E - E[j-1]), slope = (w[j] - w[j-1]) / dE
            // CDF in bin: cumW[j-1] + w[j-1]*ΔE + 0.5*slope*ΔE² = r
            // → 0.5*slope*ΔE² + w[j-1]*ΔE - (r-cumW[j-1])*total = 0
            float total = config.beamSpectrumCumWeights[nBins-1u];
            float rUn = r * total;  // un-normalize
            // first j >= 1 where cumW[j] >= rUn
            uint j = 1u;
            for (uint k = 1u; k < nBins; k++) {
                if (config.beamSpectrumCumWeights[k] >= rUn) { j = k; break; }
                j = k;
            }
            float E_lo = config.beamSpectrumEnergies[j-1u];
            float E_hi = config.beamSpectrumEnergies[j];
            float W_lo = config.beamSpectrumWeights[j-1u];
            float W_hi = config.beamSpectrumWeights[j];
            float CumLo = config.beamSpectrumCumWeights[j-1u];
            float dE = E_hi - E_lo;
            float slope = (dE > 1e-30f) ? (W_hi - W_lo) / dE : 0.0f;
            float rhs = rUn - CumLo;  // remaining integral in this bin
            float deltaE;
            if (fabs(slope) > 1e-30f) {
                // quadratic: 0.5*slope*ΔE² + W_lo*ΔE - rhs = 0
                float b = W_lo / slope;
                float c = 2.0f * rhs / slope;
                float disc = b*b + c;
                float sgn = (slope >= 0.0f) ? 1.0f : -1.0f;
                deltaE = sgn * sqrt(max(disc, 0.0f)) - b;
            } else if (W_lo > 1e-30f) {
                deltaE = rhs / W_lo;
            } else {
                deltaE = 0.0f;
            }
            photonEv = E_lo + clamp(deltaE, 0.0f, dE);
        }
    } else if (config.beamEnergySpreadEv > 0.0f) {
        // mono + Gaussian spread (G4RandGauss::shoot)
        float r1 = max(RandUniform(rng), 1e-10f);
        float r2 = RandUniform(rng);
        float mag = MOP_SQRT(-2.0f * MOP_LOG(r1));
        float dE = config.beamEnergySpreadEv * mag * MOP_COS(TWO_PI * r2);
        photonEv = max(photonEv + dE, 1e-6f);
    }

    // === Time sampling (delta or Gaussian + optional cutoff) ===
    float photonTime = gs.time;  // default 0
    if (config.beamTimeSpread > 0.0f) {
        // Rejection-sample within cutoff (cutoff 0 → no clip)
        float cutoff = config.beamTimeCutoff;
        for (int tt = 0; tt < 16; ++tt) {
            float r1 = max(RandUniform(rng), 1e-10f);
            float r2 = RandUniform(rng);
            float mag = MOP_SQRT(-2.0f * MOP_LOG(r1));
            float dt = config.beamTimeSpread * mag * MOP_COS(TWO_PI * r2);
            if (cutoff <= 0.0f || abs(dt) <= cutoff) {
                photonTime = gs.time + dt;
                break;
            }
        }
    }

    // === Photon write ===
    Photon p;
    p.position = packed_float3(genPos);
    p.direction = packed_float3(newDir);
    p.polarization = packed_float3(pol);
    p.energy = photonEv;
    p.wavelength = HC_EVNM / max(photonEv, 1e-6f);
    p.time = photonTime;
    p.weight = 1.0;
    p.status = ALIVE;
    p.volumeId = 0;

    // 2026-05-17 BEAM material auto-detect: genPos 의 cyl/box/sphere inside test.
    // gs.materialId (default 0 = World) 가 광원 위치 실제 매질과 불일치 시 잘못된 boundary
    // 처리 → 광자 lose (예: U2v5 source IN BC408 (matId=2) 인데 default Air (matId=0) 시 T=0).
    // Inner-most volume 우선 — 마지막 match 가 가장 깊은 volume (priority: cyl > box > sphere)
    uint detectedMat = gs.materialId;  // default fallback
    // Cylinder inside test (Z-aligned analytic): r²<R² && |z|<HL
    for (uint c = 0; c < numCylGeoms; c++) {
        CylinderGeometry cyl = cylinderGeometries[c];
        float3 q = genPos - float3(cyl.center);
        float r2 = q.x*q.x + q.y*q.y;
        if (r2 < cyl.radius*cyl.radius &&
            abs(q.z) < cyl.halfLength &&
            (cyl.innerRadius <= 1e-6f || r2 >= cyl.innerRadius*cyl.innerRadius)) {
            detectedMat = cyl.materialIdInside;
        }
    }
    // Box inside test
    for (uint b = 0; b < numBoxGeoms; b++) {
        BoxGeometry box = boxGeometries[b];
        float3 q = genPos - float3(box.center);
        if (abs(q.x) < box.HLX && abs(q.y) < box.HLY && abs(q.z) < box.HLZ) {
            detectedMat = box.materialIdInside;
        }
    }
    // Sphere inside test
    for (uint s = 0; s < numSphereGeoms; s++) {
        SphereGeometry sph = sphereGeometries[s];
        float3 q = genPos - float3(sph.center);
        float r2 = dot(q, q);
        if (r2 < sph.radius*sph.radius &&
            (sph.innerRadius <= 1e-6f || r2 >= sph.innerRadius*sph.innerRadius)) {
            detectedMat = sph.materialIdInside;
        }
    }
    p.materialId = detectedMat;

    p.stepCount = 0;
    p.flags = 0;
    p.reflectedCount = 0;
    p.lastHitTriId = -1;
    photons[gid] = p;

    PhotonMeta m;
    m.genstepId = gsIdx;
    m.parentTrackId = 0;
    photonMeta[gid] = m;
}
