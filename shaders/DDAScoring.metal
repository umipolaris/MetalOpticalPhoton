/**
 * DDAScoring.metal
 * Fix 65: GPU DDA ray tracing 스코어링 커널
 *
 * transit hit 배열을 입력으로 받아, 각 히트의 경로를 빈 격자를 통해 추적하고
 * 경로 길이/빈 체적을 GPU 측 빈 버퍼에 atomic 누적.
 * CPU DDA (TsScoreGPUOpticalPhotonFluence::TraceRayThroughGrid)와 동일한 로직.
 */

#include <metal_stdlib>
using namespace metal;

#include "Common.h"

// DDA 스코어링 설정 (CPU에서 전달).
// MOPTypes.h::MOPDDAConfig 와 layout 동기화 필수 (80 bytes, 16B align).
struct DDAConfig {
    float compTransX, compTransY, compTransZ;  // 컴포넌트 월드 위치 (mm)
    float compFullX, compFullY, compFullZ;      // 컴포넌트 voxel grid 폭 (mm 또는 rad)
    uint  nBinsX, nBinsY, nBinsZ;              // 빈 수 (BOX=X/Y/Z, CYL=R/Phi/Z, SPH=R/Theta/Phi)
    uint  totalTransitHits;
    float binVolume;                            // 평균/대표 bin volume (host 측 계산)
    float fluenceScale;
    uint  scoringMaterialId;
    uint  voxelType;                            // 0=BOX 1=CYLINDER 2=SPHERE
    float rMin;                                 // CYL/SPH inner radius (mm)
    float phiStart;                             // CYL Phi / SPH Phi 시작 각 (rad)
    float thetaStart;                           // SPH Theta 시작 각 (rad)
    float energyLow;                            // photon energy filter lower bound (eV), 0 = no
    float energyHigh;                           // photon energy filter upper bound (eV), 0 or large = no
    // 2026-05-20: energy binning (CPU Fluence EBins 동등). nBinsE=1 = 기존 동작.
    // binBuffer layout: [eBin * totalVoxels + voxel].
    uint  nBinsE;
    float eMinEv;
    float eMaxEv;
    uint  _pad[2];
};

// voxelType enum (0=BOX default 경로 / 1=CYLINDER / 2=SPHERE).
// enum 멤버는 unused-const-variable 경고 대상이 아니라 BOX 미참조여도 깔끔.
enum VoxelType : uint { VOXEL_BOX = 0u, VOXEL_CYLINDER = 1u, VOXEL_SPHERE = 2u };

// 2026-05-17 정석 fix: fp32 atomic_float ULP saturation 회피 — uint64 emulation.
// Apple Metal 의 atomic_ulong 미지원 → dual-uint32 (low + high with carry).
// fluence * FLUENCE_QUANT_SCALE → uint64 quantized 누적. host 가 read 시 / SCALE.
// dynamic range: max uint64 / scale = 1.8e19 / 1e6 = 1.8e13. 모든 setup 안전.
// lossless: min contrib = 0.0001 mm-2 * 1e6 = 100 (1 floor 안전).
// 2026-05-17: 1e6 → 1e10 (Cherenkov small-bin truncation fix + 안전 margin).
// 광자당 contrib = path/V_bin. large V_bin (e.g., 200³=8e6 mm³, 1×1×1 binning) 시
// contrib ~1e-5 → ×1e6 quantize = ~10 → floor truncation 5% loss per add.
// uint64 max = 1.8e19. production max bin sum (06 Mie 1B) = 8.69e8 fluence-units
//   → ×1e10 = 8.69e18 → uint64 overflow margin 2x (안전).
// scale 1e11 시 uint64 overflow 위험 — 1e10 이 안전한 production 한계.
// fp32 precision: contrib × 1e10 = ~10^6 typical, 24-bit mantissa 안 fit → 정확.
constant float FLUENCE_QUANT_SCALE = 1e10f;

inline void atomic_add_u64(device atomic_uint* low_ptr,
                           device atomic_uint* high_ptr,
                           ulong addAmount) {
    uint addLow  = (uint)(addAmount & 0xFFFFFFFFul);
    uint addHigh = (uint)(addAmount >> 32);
    // low atomic_fetch_add returns previous value → carry detection by overflow check
    uint oldLow  = atomic_fetch_add_explicit(low_ptr, addLow, memory_order_relaxed);
    uint newLow  = oldLow + addLow;  // wrap-around 동작 동일
    uint carry   = (newLow < oldLow) ? 1u : 0u;
    if (carry > 0u || addHigh > 0u) {
        atomic_fetch_add_explicit(high_ptr, addHigh + carry, memory_order_relaxed);
    }
}

kernel void ddaScoring(
    device const TransitHit*   transitHits  [[buffer(0)]],
    device const DDAConfig&    config       [[buffer(1)]],
    device atomic_uint*        binBuffer    [[buffer(2)]],  // uint64 emulation (dual-uint32 with carry). layout: [bin*2]=low,[bin*2+1]=high.
    device atomic_float*       lostValueSum [[buffer(3)]],  // fp32 lost-contrib 누적 (quantize 0 lost). host 가 binBuffer total 대비 비율로 판정
    uint gid [[thread_position_in_grid]])
{
    if (gid >= config.totalTransitHits) return;

    TransitHit hit = transitHits[gid];

    // F2 정석 fix (2026-04-23): scorer material 필터.
    // 0xFFFFFFFF = all accept (호환성). 그 외엔 hit.materialId 일치만 누적.
    if (config.scoringMaterialId != 0xFFFFFFFFu &&
        hit.materialId != config.scoringMaterialId) return;

    // 2026-05-19: photon energy filter (multi-scorer support).
    if (config.energyLow > 0.0f && hit.energy < config.energyLow) return;
    if (config.energyHigh > 0.0f && hit.energy > config.energyHigh) return;

    // transit hit 디코딩: weight = geometric 경로 길이 (mm), photonWeight = 광자 가중치
    float pathLength = hit.weight;
    float photonWeight = hit.photonWeight;
    if (pathLength < 0.001f) return;  // 1μm 미만 무시

    float3 startPos = float3(hit.position);
    float3 dir = float3(hit.direction);

    // 방향 정규화
    float dirLen = length(dir);
    if (dirLen < 1e-10f) return;
    dir /= dirLen;

    // 2026-05-20: energy → E bin offset. binBuffer layout = [eBin*totalVoxels + voxel].
    // nBinsE=1 (default) 시 eBinOffset=0 → 기존 동작 (regression 없음).
    // 광자 1개당 energy 고정 → eBinOffset 도 고정 (모든 voxel 동일 eBin).
    uint eBinOffset = 0u;
    if (config.nBinsE > 1u) {
        float ef = (hit.energy - config.eMinEv) / (config.eMaxEv - config.eMinEv);
        int eb = clamp(int(ef * float(config.nBinsE)), 0, int(config.nBinsE) - 1);
        eBinOffset = uint(eb) * (config.nBinsX * config.nBinsY * config.nBinsZ);
    }

    // ----------------------------------------------------------------
    // 2026-05-16 Phase 1.3 (sphere analytic shell): SPHERE case 의 sample
    // 범위를 ray ∩ (outer sphere) ∩ (inner sphere) ∩ (hemisphere clip) 의
    // valid t range 만 사용. sample density 자연히 충분 (thin shell 도
    // 정확). cylinder case 는 기존 ray pathLength 전체 sample 그대로.
    // ----------------------------------------------------------------
    if (config.voxelType == VOXEL_SPHERE) {
        // ray-sphere shell analytic intersection
        float3 origin = startPos - float3(config.compTransX, config.compTransY, config.compTransZ);
        float3 D = dir;
        float RMax = config.rMin + config.compFullX;
        float c_o = dot(origin, origin) - RMax*RMax;
        float b_half = dot(D, origin);
        float disc_o = b_half*b_half - c_o;
        if (disc_o < 0.0f) return;  // ray miss outer sphere
        float sq_o = sqrt(disc_o);
        float t_outer_in  = -b_half - sq_o;
        float t_outer_out = -b_half + sq_o;

        // inner sphere
        float t_inner_in = 1e20f, t_inner_out = -1e20f;
        bool hasInner = (config.rMin > 1e-6f);
        if (hasInner) {
            float c_i = dot(origin, origin) - config.rMin * config.rMin;
            float disc_i = b_half*b_half - c_i;
            if (disc_i >= 0.0f) {
                float sq_i = sqrt(disc_i);
                t_inner_in  = -b_half - sq_i;
                t_inner_out = -b_half + sq_i;
            }
        }

        // hemisphere theta clip: thetaStart=0 (upper, z>=0) 또는 thetaStart=π/2 (lower, z<=0).
        // full sphere (deltaTheta≈π) 시 clip 없음. 일반 theta range 는 미지원 (cone intersection 필요).
        float t_clip_min = -1e20f, t_clip_max = 1e20f;
        bool isUpper = (config.thetaStart < 0.01f) && (config.compFullZ > 1.56f && config.compFullZ < 1.58f);
        bool isLower = (config.thetaStart > 1.56f && config.thetaStart < 1.58f) && (config.compFullZ > 1.56f && config.compFullZ < 1.58f);
        if (isUpper) {
            if (abs(D.z) > 1e-10f) {
                float t_z = -origin.z / D.z;
                if (D.z > 0.0f) t_clip_min = max(t_clip_min, t_z);
                else            t_clip_max = min(t_clip_max, t_z);
            } else if (origin.z < 0.0f) return;
        } else if (isLower) {
            if (abs(D.z) > 1e-10f) {
                float t_z = -origin.z / D.z;
                if (D.z < 0.0f) t_clip_min = max(t_clip_min, t_z);
                else            t_clip_max = min(t_clip_max, t_z);
            } else if (origin.z > 0.0f) return;
        }

        // 전체 valid range = ray ∩ outer sphere ∩ hemisphere clip ∩ [0, pathLength]
        float t_lo = max(max(t_outer_in, t_clip_min), 0.0f);
        float t_hi = min(min(t_outer_out, t_clip_max), pathLength);
        if (t_lo >= t_hi) return;

        // shell 내부 segments: [t_lo, t_hi] 에서 inner sphere 제외
        // up to 2 segments
        float segs[4];  // [seg1_in, seg1_out, seg2_in, seg2_out]
        int nSegs = 0;
        if (!hasInner || t_inner_in >= t_hi || t_inner_out <= t_lo) {
            segs[0] = t_lo; segs[1] = t_hi; nSegs = 1;
        } else {
            float s1_in  = t_lo;
            float s1_out = min(t_inner_in, t_hi);
            if (s1_in < s1_out) { segs[nSegs*2] = s1_in; segs[nSegs*2+1] = s1_out; nSegs++; }
            float s2_in  = max(t_inner_out, t_lo);
            float s2_out = t_hi;
            if (s2_in < s2_out) { segs[nSegs*2] = s2_in; segs[nSegs*2+1] = s2_out; nSegs++; }
        }
        if (nSegs == 0) return;

        const float TWO_PI = 6.28318530717958647692f;
        float binStep0 = config.compFullX / float(config.nBinsX);
        float binStep1 = config.compFullY / float(config.nBinsY);
        float binStep2 = config.compFullZ / float(config.nBinsZ);

        uint  curBinIndex = 0xFFFFFFFFu;
        float curAccum    = 0.0f;

        const int N = 128;  // sample per segment — segment 가 shell 안만이라 적은 N 으로도 충분
        for (int sg = 0; sg < nSegs; sg++) {
            float t_in  = segs[sg*2];
            float t_out = segs[sg*2 + 1];
            float segLen = t_out - t_in;
            float dseg = segLen / float(N);
            float baseContrib = dseg * photonWeight * config.fluenceScale;
            if (baseContrib <= 0.0f) continue;

            for (int s = 0; s <= N; s++) {
                uint binIndex = 0xFFFFFFFFu;
                float contrib = 0.0f;

                if (s < N) {
                    float t = t_in + (float(s) + 0.5f) * dseg;
                    float3 p = origin + D * t;
                    float r = sqrt(dot(p, p));
                    if (r >= config.rMin && r < config.rMin + config.compFullX) {
                        float theta = acos(clamp(p.z / max(r, 1e-30f), -1.0f, 1.0f))
                                    - config.thetaStart;
                        if (theta >= 0.0f && theta < float(config.nBinsZ) * binStep2) {
                            int i0 = clamp(int((r - config.rMin) / binStep0),
                                           0, int(config.nBinsX) - 1);
                            int i1 = 0;
                            if (config.nBinsY > 1u) {
                                float phi = atan2(p.y, p.x) - config.phiStart;
                                phi = fmod(phi, TWO_PI);
                                if (phi < 0.0f) phi += TWO_PI;
                                i1 = clamp(int(phi / binStep1),
                                           0, int(config.nBinsY) - 1);
                            }
                            int i2 = clamp(int(theta / binStep2),
                                           0, int(config.nBinsZ) - 1);
                            float rIn  = config.rMin + float(i0) * binStep0;
                            float rOut = rIn + binStep0;
                            float thIn  = config.thetaStart + float(i2)     * binStep2;
                            float thOut = config.thetaStart + float(i2 + 1) * binStep2;
                            float binVol = (rOut*rOut*rOut - rIn*rIn*rIn) / 3.0f
                                         * (cos(thIn) - cos(thOut)) * binStep1;
                            if (binVol > 1e-30f) {
                                binIndex = uint(i0) * config.nBinsY * config.nBinsZ
                                         + uint(i1) * config.nBinsZ + uint(i2);
                                contrib = baseContrib / binVol;
                            }
                        }
                    }
                }

                if (binIndex == curBinIndex && binIndex != 0xFFFFFFFFu) {
                    curAccum += contrib;
                } else {
                    if (curBinIndex != 0xFFFFFFFFu && curAccum > 0.0f) {
                        { ulong aq = (ulong)(curAccum * FLUENCE_QUANT_SCALE); if (aq == 0ul) { atomic_fetch_add_explicit(lostValueSum, curAccum, memory_order_relaxed); } else { atomic_add_u64(&binBuffer[(curBinIndex+eBinOffset)*2u], &binBuffer[(curBinIndex+eBinOffset)*2u + 1u], aq); } }
                    }
                    curBinIndex = binIndex;
                    curAccum    = contrib;
                }
            }
            // segment 끝 시 flush
            if (curBinIndex != 0xFFFFFFFFu && curAccum > 0.0f) {
                { ulong aq = (ulong)(curAccum * FLUENCE_QUANT_SCALE); if (aq == 0ul) { atomic_fetch_add_explicit(lostValueSum, curAccum, memory_order_relaxed); } else { atomic_add_u64(&binBuffer[(curBinIndex+eBinOffset)*2u], &binBuffer[(curBinIndex+eBinOffset)*2u + 1u], aq); } }
                curBinIndex = 0xFFFFFFFFu;
                curAccum = 0.0f;
            }
        }
        return;
    }

    // 2026-05-17 정석 fix: CYLINDER analytic boundary-crossing DDA.
    // 이전 sample-based (N=256) 가 ray 의 axis-aligned 방향에서 atan2 quantization
    // 4-fold artifact (φ=0/90/180/270° peak, k=4 amp 0.13%, CV 0.11%) 발생 →
    // ray 의 R / φ / Z bin boundary crossings 를 analytic 으로 정확히 분할 후 segment
    // 별 contribution. N 무관 → 항상 정확.
    //
    // boundary crossings:
    //   R: r²(t) = R_b² → quadratic in t (a t² + b t + c0 - R² = 0)
    //   φ: tan(phi_b) = ry/rx → rx*sin(phi_b) - ry*cos(phi_b) = 0 → linear in t
    //   Z: z(t) = z_b → linear in t
    if (config.voxelType == VOXEL_CYLINDER) {
        const float TWO_PI = 6.28318530717958647692f;
        float halfZ_cyl = config.compFullZ * 0.5f;
        float binStep0 = config.compFullX / float(config.nBinsX);
        float binStep1 = config.compFullY / float(config.nBinsY);
        float binStep2 = config.compFullZ / float(config.nBinsZ);

        // Local coordinate (component-relative)
        float sx0 = startPos.x - config.compTransX;
        float sy0 = startPos.y - config.compTransY;
        float sz0 = startPos.z - config.compTransZ;
        float dxL = dir.x;
        float dyL = dir.y;
        float dzL = dir.z;

        float baseContrib = photonWeight * config.fluenceScale;  // per-unit-length
        if (baseContrib <= 0.0f) return;

        // Candidate crossing t 수집. max 512 (RBins+PhiBins+ZBins + entry/exit + safety).
        const int MAX_T = 512;
        float ts[MAX_T];
        int nT = 0;
        ts[nT++] = 0.0f;
        ts[nT++] = pathLength;

        // R bin crossings (RBins>1 시)
        if (config.nBinsX > 1u) {
            float a = dxL*dxL + dyL*dyL;
            float b = 2.0f * (sx0*dxL + sy0*dyL);
            float c0 = sx0*sx0 + sy0*sy0;
            if (a > 1e-30f) {
                for (uint ir = 1u; ir < config.nBinsX && nT < MAX_T - 1; ir++) {
                    float R = config.rMin + float(ir) * binStep0;
                    float disc = b*b - 4.0f*a*(c0 - R*R);
                    if (disc > 0.0f) {
                        float sd = sqrt(disc);
                        float t1 = (-b - sd) / (2.0f * a);
                        float t2 = (-b + sd) / (2.0f * a);
                        if (t1 > 1e-7f && t1 < pathLength - 1e-7f && nT < MAX_T) ts[nT++] = t1;
                        if (t2 > 1e-7f && t2 < pathLength - 1e-7f && nT < MAX_T) ts[nT++] = t2;
                    }
                }
            }
        }
        // Phi bin crossings (PhiBins>1 시)
        if (config.nBinsY > 1u) {
            for (uint ip = 0u; ip < config.nBinsY && nT < MAX_T - 1; ip++) {
                float phi_b = config.phiStart + float(ip) * binStep1;
                float sb = MOP_SIN(phi_b);
                float cb = MOP_COS(phi_b);
                float denom = dxL*sb - dyL*cb;
                if (abs(denom) > 1e-20f) {
                    float t = -(sx0*sb - sy0*cb) / denom;
                    if (t > 1e-7f && t < pathLength - 1e-7f && nT < MAX_T) ts[nT++] = t;
                }
            }
        }
        // Z bin crossings (ZBins>1 시)
        if (config.nBinsZ > 1u && abs(dzL) > 1e-20f) {
            for (uint iz = 1u; iz < config.nBinsZ && nT < MAX_T - 1; iz++) {
                float z_b = -halfZ_cyl + float(iz) * binStep2;
                float t = (z_b - sz0) / dzL;
                if (t > 1e-7f && t < pathLength - 1e-7f && nT < MAX_T) ts[nT++] = t;
            }
        }

        // Insertion sort (small nT, typically < 20 valid crossings).
        for (int i = 1; i < nT; i++) {
            float v = ts[i];
            int j = i - 1;
            while (j >= 0 && ts[j] > v) {
                ts[j+1] = ts[j];
                j--;
            }
            ts[j+1] = v;
        }

        // Process consecutive segments.
        uint  curBinIndex = 0xFFFFFFFFu;
        float curAccum    = 0.0f;
        for (int k = 0; k < nT - 1; k++) {
            float t0 = ts[k];
            float t1 = ts[k+1];
            float segLen = t1 - t0;
            if (segLen < 1e-9f) continue;
            float t_mid = (t0 + t1) * 0.5f;
            float rx = sx0 + dxL * t_mid;
            float ry = sy0 + dyL * t_mid;
            float rz = sz0 + dzL * t_mid;

            uint binIndex = 0xFFFFFFFFu;
            float contrib = 0.0f;
            float rxy = sqrt(rx*rx + ry*ry);
            if (rxy >= config.rMin && rxy < config.rMin + config.compFullX
                && abs(rz) < halfZ_cyl) {
                int i0 = clamp(int((rxy - config.rMin) / binStep0),
                               0, int(config.nBinsX) - 1);
                int i1 = 0;
                if (config.nBinsY > 1u) {
                    float phi = atan2(ry, rx) - config.phiStart;
                    phi = fmod(phi, TWO_PI);
                    if (phi < 0.0f) phi += TWO_PI;
                    i1 = clamp(int(phi / binStep1),
                               0, int(config.nBinsY) - 1);
                }
                int i2 = clamp(int((rz + halfZ_cyl) / binStep2),
                               0, int(config.nBinsZ) - 1);
                float rIn  = config.rMin + float(i0) * binStep0;
                float rOut = rIn + binStep0;
                float binVol = (rOut*rOut - rIn*rIn) * 0.5f * binStep1 * binStep2;
                if (binVol > 1e-30f) {
                    binIndex = uint(i0) * config.nBinsY * config.nBinsZ
                             + uint(i1) * config.nBinsZ + uint(i2);
                    contrib = segLen * baseContrib / binVol;
                }
            }

            if (binIndex == curBinIndex && binIndex != 0xFFFFFFFFu) {
                curAccum += contrib;
            } else {
                if (curBinIndex != 0xFFFFFFFFu && curAccum > 0.0f) {
                    { ulong aq = (ulong)(curAccum * FLUENCE_QUANT_SCALE); if (aq == 0ul) { atomic_fetch_add_explicit(lostValueSum, curAccum, memory_order_relaxed); } else { atomic_add_u64(&binBuffer[(curBinIndex+eBinOffset)*2u], &binBuffer[(curBinIndex+eBinOffset)*2u + 1u], aq); } }
                }
                curBinIndex = binIndex;
                curAccum    = contrib;
            }
        }
        if (curBinIndex != 0xFFFFFFFFu && curAccum > 0.0f) {
            { ulong aq = (ulong)(curAccum * FLUENCE_QUANT_SCALE); if (aq == 0ul) { atomic_fetch_add_explicit(lostValueSum, curAccum, memory_order_relaxed); } else { atomic_add_u64(&binBuffer[(curBinIndex+eBinOffset)*2u], &binBuffer[(curBinIndex+eBinOffset)*2u + 1u], aq); } }
        }
        return;
    }

    // ---------------- BOX path (기존 Cartesian DDA) ----------------
    // 컴포넌트 로컬 좌표
    float localX = startPos.x - config.compTransX;
    float localY = startPos.y - config.compTransY;
    float localZ = startPos.z - config.compTransZ;

    float halfX = config.compFullX * 0.5f;
    float halfY = config.compFullY * 0.5f;
    float halfZ = config.compFullZ * 0.5f;

    float binSizeX = config.compFullX / float(config.nBinsX);
    float binSizeY = config.compFullY / float(config.nBinsY);
    float binSizeZ = config.compFullZ / float(config.nBinsZ);

    // 광선-AABB 교차
    float tmin = 0.0f, tmax = pathLength;

    // X축
    if (abs(dir.x) > 1e-10f) {
        float t1 = (-halfX - localX) / dir.x;
        float t2 = ( halfX - localX) / dir.x;
        if (t1 > t2) { float tmp = t1; t1 = t2; t2 = tmp; }
        tmin = max(tmin, t1);
        tmax = min(tmax, t2);
    } else {
        if (localX < -halfX || localX > halfX) return;
    }

    // Y축
    if (abs(dir.y) > 1e-10f) {
        float t1 = (-halfY - localY) / dir.y;
        float t2 = ( halfY - localY) / dir.y;
        if (t1 > t2) { float tmp = t1; t1 = t2; t2 = tmp; }
        tmin = max(tmin, t1);
        tmax = min(tmax, t2);
    } else {
        if (localY < -halfY || localY > halfY) return;
    }

    // Z축
    if (abs(dir.z) > 1e-10f) {
        float t1 = (-halfZ - localZ) / dir.z;
        float t2 = ( halfZ - localZ) / dir.z;
        if (t1 > t2) { float tmp = t1; t1 = t2; t2 = tmp; }
        tmin = max(tmin, t1);
        tmax = min(tmax, t2);
    } else {
        if (localZ < -halfZ || localZ > halfZ) return;
    }

    if (tmin >= tmax) return;

    // 진입점 빈 인덱스
    float entryX = localX + dir.x * tmin;
    float entryY = localY + dir.y * tmin;
    float entryZ = localZ + dir.z * tmin;

    int iX = int((entryX + halfX) / binSizeX);
    int iY = int((entryY + halfY) / binSizeY);
    int iZ = int((entryZ + halfZ) / binSizeZ);

    iX = clamp(iX, 0, int(config.nBinsX) - 1);
    iY = clamp(iY, 0, int(config.nBinsY) - 1);
    iZ = clamp(iZ, 0, int(config.nBinsZ) - 1);

    // DDA 스텝 방향
    int stepIX = (dir.x >= 0) ? 1 : -1;
    int stepIY = (dir.y >= 0) ? 1 : -1;
    int stepIZ = (dir.z >= 0) ? 1 : -1;

    float tMaxX, tMaxY, tMaxZ;
    float tDeltaX, tDeltaY, tDeltaZ;

    if (abs(dir.x) > 1e-10f) {
        float nextBoundX = -halfX + float(iX + (stepIX > 0 ? 1 : 0)) * binSizeX;
        tMaxX = tmin + (nextBoundX - entryX) / dir.x;
        tDeltaX = binSizeX / abs(dir.x);
    } else {
        tMaxX = 1e20f;
        tDeltaX = 1e20f;
    }

    if (abs(dir.y) > 1e-10f) {
        float nextBoundY = -halfY + float(iY + (stepIY > 0 ? 1 : 0)) * binSizeY;
        tMaxY = tmin + (nextBoundY - entryY) / dir.y;
        tDeltaY = binSizeY / abs(dir.y);
    } else {
        tMaxY = 1e20f;
        tDeltaY = 1e20f;
    }

    if (abs(dir.z) > 1e-10f) {
        float nextBoundZ = -halfZ + float(iZ + (stepIZ > 0 ? 1 : 0)) * binSizeZ;
        tMaxZ = tmin + (nextBoundZ - entryZ) / dir.z;
        tDeltaZ = binSizeZ / abs(dir.z);
    } else {
        tMaxZ = 1e20f;
        tDeltaZ = 1e20f;
    }

    // DDA 순회
    float tCurrent = tmin;
    int maxSteps = int(config.nBinsX + config.nBinsY + config.nBinsZ) + 10;

    for (int s = 0; s < maxSteps; s++) {
        if (iX < 0 || iX >= int(config.nBinsX) ||
            iY < 0 || iY >= int(config.nBinsY) ||
            iZ < 0 || iZ >= int(config.nBinsZ)) break;

        float tNext = min(min(tMaxX, tMaxY), min(tMaxZ, tmax));
        float segLen = tNext - tCurrent;

        if (segLen > 0.0f) {
            // binIndex = iX*nY*nZ + iY*nZ + iZ  (TOPAS CPU Fluence 호환: Z-fastest)
            uint binIndex = uint(iX) * config.nBinsY * config.nBinsZ + uint(iY) * config.nBinsZ + uint(iZ);

            // 2026-05-12 fix: atomic_uint (uint32, max 4.3e9) → atomic_float (fp32)
            // production 1B × focal spot 시 uint32 max 4.3e9 cumulative overflow 발생.
            // Apple Metal atomic_ulong fetch_add 미지원 → atomic_float 사용.
            // fluenceScale 은 host 측에서 1.0f 설정 (float 누적 시 scaling 불필요).
            // fp32 mantissa 24-bit → 누적값 1e7 까지 precision ~1, 1.6e6 peak 까지 안전.
            float contrib_f = segLen * photonWeight * config.fluenceScale / config.binVolume;
            if (contrib_f > 0.0f) {
                // uint64 emulation (dual-uint32 carry): 1B focal-spot fp32 overflow 방지.
                ulong aq = (ulong)(contrib_f * FLUENCE_QUANT_SCALE);
                if (aq == 0ul) {
                    atomic_fetch_add_explicit(lostValueSum, contrib_f, memory_order_relaxed);
                } else {
                    atomic_add_u64(&binBuffer[(binIndex+eBinOffset)*2u],
                                   &binBuffer[(binIndex+eBinOffset)*2u + 1u], aq);
                }
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
