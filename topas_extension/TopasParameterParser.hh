/**
 * TopasParameterParser.hh
 * TOPAS .topas 파라미터 파일에서 광학 속성을 파싱
 *
 * TOPAS 파라미터 문법:
 *   dv:Ma/MaterialName/Property/Energies = N val1 val2 ... unit
 *   uv:Ma/MaterialName/Property/Values   = N val1 val2 ...
 *   u:Ma/MaterialName/ScalarProperty     = value
 *   d:Ma/MaterialName/ScalarProperty     = value unit
 *   s:Su/SurfaceName/Property            = "string"
 *   s:Ge/CompName/OpticalBehaviorTo/CompName2 = "SurfaceName"
 *   s:Ge/CompName/OpticalBehavior              = "SurfaceName"
 */

#ifndef TOPAS_PARAMETER_PARSER_HH
#define TOPAS_PARAMETER_PARSER_HH

#include "MOPTypes.hh"
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <cstring>

// ============================================================
// 파싱된 물질 정보
// ============================================================
struct ParsedMaterial {
    std::string name;
    MOPMaterialProperties props;

    ParsedMaterial() {
        memset(&props, 0, sizeof(props));
        props.resolutionScale = 1.0f;  // 기본값
        props.yieldRatio = 1.0f;
    }
};

// ============================================================
// 파싱된 표면 정보
// ============================================================
struct ParsedSurface {
    std::string name;
    MOPSurfaceProperties props;

    ParsedSurface() {
        memset(&props, 0, sizeof(props));
    }
};

// ============================================================
// 볼륨 간 표면 바인딩 정보
// ============================================================
struct ParsedSurfaceBinding {
    std::string volumeName1;
    std::string volumeName2;  // 비어있으면 skin surface
    std::string surfaceName;
    bool isSkinSurface;
};

// ============================================================
// TopasParameterParser
// ============================================================
class TopasParameterParser {
public:
    TopasParameterParser();
    ~TopasParameterParser();

    /**
     * TOPAS 파라미터 파일 파싱
     * includeFile 지시자도 재귀적으로 처리
     *
     * @param filePath .topas 파일 경로
     * @return 0=성공, -1=파일 열기 실패, -2=파싱 오류
     */
    int ParseFile(const std::string& filePath);

    /**
     * 여러 파일을 순서대로 파싱 (later overrides earlier)
     */
    int ParseFiles(const std::vector<std::string>& filePaths);

    // === 결과 접근 ===
    const std::vector<ParsedMaterial>& GetMaterials() const { return m_materials; }
    const std::vector<ParsedSurface>& GetSurfaces() const { return m_surfaces; }
    const std::vector<ParsedSurfaceBinding>& GetSurfaceBindings() const { return m_bindings; }

    /**
     * 이름으로 물질 검색
     * @return 물질 인덱스, 없으면 -1
     */
    int FindMaterialByName(const std::string& name) const;

    /**
     * 이름으로 표면 검색
     * @return 표면 인덱스, 없으면 -1
     */
    int FindSurfaceByName(const std::string& name) const;

    /**
     * 물리 프로세스 활성화 상태
     */
    bool IsScintillationEnabled() const { return m_scintillationEnabled; }
    bool IsCerenkovEnabled() const { return m_cerenkovEnabled; }
    bool IsOpticalAbsorptionEnabled() const { return m_absorptionEnabled; }
    bool IsRayleighEnabled() const { return m_rayleighEnabled; }
    bool IsMieEnabled() const { return m_mieEnabled; }
    bool IsWLSEnabled() const { return m_wlsEnabled; }
    bool IsBoundaryEnabled() const { return m_boundaryEnabled; }

    // 체렌코프 설정
    int    GetCerenkovMaxPhotonsPerStep() const { return m_cerenkovMaxPhotons; }
    float  GetCerenkovMaxBetaChange() const { return m_cerenkovMaxBetaChange; }

    // 2026-05-18: AllowMT — default false (hard guard).
    // True 시 worker thread MT 허용 (race 보호용 m_gpuMutex 안에서 직렬화).
    bool   GetGPUOpticalAllowMT() const { return m_gpuOpticalAllowMT; }

private:
    // 파싱 내부 함수
    int ParseLine(const std::string& line, const std::string& baseDir);
    void ParseMaterialProperty(const std::string& matName,
                                const std::string& propPath,
                                const std::string& typePrefix,
                                const std::string& valueStr);
    void ParseSurfaceProperty(const std::string& surfName,
                               const std::string& propName,
                               const std::string& typePrefix,
                               const std::string& valueStr);
    void ParseGeometryBinding(const std::string& compName,
                               const std::string& propPath,
                               const std::string& valueStr);
    void ParsePhysicsSetting(const std::string& key, const std::string& value);

    // 값 파싱 헬퍼
    std::vector<float> ParseFloatVector(const std::string& str, int expectedCount);
    float ParseFloat(const std::string& str);
    std::string ParseString(const std::string& str);
    float ConvertEnergyUnit(float value, const std::string& unit);
    float ConvertTimeUnit(float value, const std::string& unit);
    float ConvertLengthUnit(float value, const std::string& unit);

    // 물질/표면 찾기 또는 생성
    ParsedMaterial& GetOrCreateMaterial(const std::string& name);
    ParsedSurface& GetOrCreateSurface(const std::string& name);

    void FillPropertyTable(MOPPropertyTable& table,
                           const std::vector<float>& energies,
                           const std::vector<float>& values);

    // 데이터 저장
    std::vector<ParsedMaterial> m_materials;
    std::vector<ParsedSurface> m_surfaces;
    std::vector<ParsedSurfaceBinding> m_bindings;
    std::unordered_map<std::string, int> m_materialNameMap;
    std::unordered_map<std::string, int> m_surfaceNameMap;

    // 임시 버퍼: 에너지 벡터 (Energies와 Values가 별도 줄)
    struct PendingVector {
        std::vector<float> energies;
        std::string unit;
    };
    std::map<std::string, PendingVector> m_pendingEnergies;

    // 물리 프로세스 활성화 상태
    bool m_scintillationEnabled;
    bool m_cerenkovEnabled;
    bool m_absorptionEnabled;
    bool m_rayleighEnabled;
    bool m_mieEnabled;
    bool m_wlsEnabled;
    bool m_boundaryEnabled;

    int   m_cerenkovMaxPhotons;
    float m_cerenkovMaxBetaChange;
    bool  m_gpuOpticalAllowMT;

    // 파싱된 파일 추적 (중복 include 방지)
    std::vector<std::string> m_parsedFiles;
};

#endif /* TOPAS_PARAMETER_PARSER_HH */
