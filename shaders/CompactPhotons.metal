/**
 * CompactPhotons.metal
 * Fix 66: 광자 컴팩션 커널 — 사망 광자 제거, 살아있는 광자만 재배치
 *
 * 2-pass 알고리즘:
 *   Pass 1 (countAlive):  각 광자의 alive 여부 → prefix sum으로 새 인덱스 계산
 *   Pass 2 (compact):     살아있는 광자를 새 인덱스로 복사
 *
 * 간소화 버전: threadgroup 단위 카운트 + atomic 전역 카운터
 */

#include <metal_stdlib>
using namespace metal;

#include "Common.h"

// Pass 1: 살아있는 광자 수 카운트 + 각 광자의 compact 대상 인덱스 할당
kernel void compactPhotons(
    device Photon*       photonsIn   [[buffer(0)]],  // 입력 (현재)
    device Photon*       photonsOut  [[buffer(1)]],  // 출력 (compact 결과)
    device atomic_uint&  aliveCount  [[buffer(2)]],  // 살아있는 광자 수
    device const uint&   totalPhotons [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= totalPhotons) return;

    Photon p = photonsIn[gid];

    if (p.status == ALIVE) {
        // 살아있는 광자: atomic으로 새 인덱스 할당
        uint newIdx = atomic_fetch_add_explicit(&aliveCount, 1, memory_order_relaxed);
        photonsOut[newIdx] = p;
    }
}
