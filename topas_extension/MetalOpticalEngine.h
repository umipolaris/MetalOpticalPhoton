/**
 * MetalOpticalEngine.h — TOPAS Extension wrapper
 * Canonical source: include/MetalOpticalEngine.h
 *   (전체 C API: Create/Propagate + geometry + DDA + multi-scorer)
 *
 * 과거엔 이 파일이 include/ 의 별도 사본이었는데, 같은 include guard
 * (METAL_OPTICAL_ENGINE_H) 를 공유하면서 내용이 갈라져 있었다 — 한쪽엔 geometry
 * API 만, 다른 쪽엔 multi-scorer API 만 있었고 RegisterScorerBinBuffer 시그니처도
 * 실제 dylib(.mm) 와 어긋나 있었다(2-arg vs 5-arg). 두 헤더가 같은 가드라 어느
 * translation unit 이 어느 쪽을 먼저 include 하느냐에 따라 보이는 API 가 달라지는
 * 배포 지뢰였다. 이제 canonical 헤더를 그대로 포함하는 wrapper 로 단일화한다.
 */
#include "../include/MetalOpticalEngine.h"
