// Extra Class for TopasParameterParser
/**
 * TopasParameterParser.cc
 * TOPAS .topas 파라미터 파일에서 광학 속성을 파싱하는 구현
 */

#include "TopasParameterParser.hh"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <filesystem>

namespace fs = std::filesystem;

// ============================================================
// 문자열 유틸리티
// ============================================================
static std::string Trim(const std::string& s) {
    auto start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

static std::string ToLower(const std::string& s) {
    std::string out = s;
    std::transform(out.begin(), out.end(), out.begin(), ::tolower);
    return out;
}

static std::vector<std::string> Split(const std::string& s, char delim) {
    std::vector<std::string> tokens;
    std::istringstream iss(s);
    std::string tok;
    while (std::getline(iss, tok, delim)) {
        std::string trimmed = Trim(tok);
        if (!trimmed.empty()) tokens.push_back(trimmed);
    }
    return tokens;
}

// ============================================================
// 생성자/소멸자
// ============================================================
TopasParameterParser::TopasParameterParser()
    : m_scintillationEnabled(true)
    , m_cerenkovEnabled(true)
    , m_absorptionEnabled(true)
    , m_rayleighEnabled(true)
    , m_mieEnabled(true)
    , m_wlsEnabled(true)
    , m_boundaryEnabled(true)
    , m_cerenkovMaxPhotons(100)
    , m_cerenkovMaxBetaChange(10.0f)
    , m_gpuOpticalAllowMT(false)
{
}

TopasParameterParser::~TopasParameterParser() {}

// ============================================================
// 단위 변환
// ============================================================
float TopasParameterParser::ConvertEnergyUnit(float value, const std::string& unit) {
    std::string u = ToLower(unit);
    if (u == "ev")  return value;
    if (u == "kev") return value * 1000.0f;
    if (u == "mev") return value * 1.0e6f;
    if (u == "gev") return value * 1.0e9f;
    // nm -> eV 변환: E(eV) = 1239.84198 / λ(nm)
    if (u == "nm")  return 1239.84198f / value;
    return value;  // 기본: eV
}

float TopasParameterParser::ConvertTimeUnit(float value, const std::string& unit) {
    std::string u = ToLower(unit);
    if (u == "ns")  return value;
    if (u == "us")  return value * 1000.0f;
    if (u == "ms")  return value * 1.0e6f;
    if (u == "s")   return value * 1.0e9f;
    if (u == "ps")  return value * 0.001f;
    return value;
}

float TopasParameterParser::ConvertLengthUnit(float value, const std::string& unit) {
    std::string u = ToLower(unit);
    if (u == "mm")  return value;
    if (u == "cm")  return value * 10.0f;
    if (u == "m")   return value * 1000.0f;
    if (u == "um")  return value * 0.001f;
    return value;
}

// ============================================================
// 파일 파싱
// ============================================================
int TopasParameterParser::ParseFile(const std::string& filePath) {
    // 이미 파싱한 파일인지 확인
    for (const auto& f : m_parsedFiles) {
        if (f == filePath) return 0;
    }

    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "[MOP] Error: Cannot open file: " << filePath << std::endl;
        return -1;
    }

    m_parsedFiles.push_back(filePath);
    std::string baseDir = fs::path(filePath).parent_path().string();

    std::string line;
    int lineNum = 0;
    while (std::getline(file, line)) {
        lineNum++;
        line = Trim(line);

        // 빈 줄, 주석 건너뛰기
        if (line.empty() || line[0] == '#') continue;

        // 인라인 주석 제거
        auto commentPos = line.find('#');
        if (commentPos != std::string::npos) {
            line = Trim(line.substr(0, commentPos));
        }

        // includeFile 처리
        if (line.find("includeFile") == 0) {
            auto eqPos = line.find('=');
            if (eqPos != std::string::npos) {
                std::string includePath = Trim(line.substr(eqPos + 1));
                // 따옴표 제거
                if (!includePath.empty() && includePath.front() == '"')
                    includePath = includePath.substr(1);
                if (!includePath.empty() && includePath.back() == '"')
                    includePath.pop_back();

                // 상대 경로 → 절대 경로
                if (!fs::path(includePath).is_absolute()) {
                    includePath = baseDir + "/" + includePath;
                }

                int ret = ParseFile(includePath);
                if (ret != 0) {
                    std::cerr << "[MOP] Warning: Failed to include file: "
                              << includePath << " (line " << lineNum << ")" << std::endl;
                }
            }
            continue;
        }

        int ret = ParseLine(line, baseDir);
        if (ret != 0) {
            std::cerr << "[MOP] Warning: Parse error at line " << lineNum
                      << ": " << line << std::endl;
        }
    }

    return 0;
}

int TopasParameterParser::ParseFiles(const std::vector<std::string>& filePaths) {
    for (const auto& path : filePaths) {
        int ret = ParseFile(path);
        if (ret != 0) return ret;
    }
    return 0;
}

// ============================================================
// 한 줄 파싱
// ============================================================
int TopasParameterParser::ParseLine(const std::string& line, const std::string& /*baseDir*/) {
    // TOPAS 파라미터 형식: type:Category/Name/.../Property = value
    auto colonPos = line.find(':');
    if (colonPos == std::string::npos || colonPos == 0) return 0;

    auto eqPos = line.find('=');
    if (eqPos == std::string::npos) return 0;

    std::string typePrefix = Trim(line.substr(0, colonPos));   // "dv", "uv", "u", "d", "s", "b", etc.
    std::string key = Trim(line.substr(colonPos + 1, eqPos - colonPos - 1));
    std::string value = Trim(line.substr(eqPos + 1));

    // 카테고리 분기
    if (key.substr(0, 3) == "Ma/") {
        // === 물질 속성 ===
        // Ma/MaterialName/Property 또는 Ma/MaterialName/Property/Energies
        auto parts = Split(key, '/');
        if (parts.size() < 3) return 0;

        std::string matName = parts[1];
        std::string propPath;
        for (size_t i = 2; i < parts.size(); i++) {
            if (i > 2) propPath += "/";
            propPath += parts[i];
        }

        ParseMaterialProperty(matName, propPath, typePrefix, value);
    }
    else if (key.substr(0, 3) == "Su/") {
        // === 표면 속성 ===
        auto parts = Split(key, '/');
        if (parts.size() < 3) return 0;

        std::string surfName = parts[1];
        std::string propName;
        for (size_t i = 2; i < parts.size(); i++) {
            if (i > 2) propName += "/";
            propName += parts[i];
        }

        ParseSurfaceProperty(surfName, propName, typePrefix, value);
    }
    else if (key.substr(0, 3) == "Ge/") {
        // === 지오메트리 광학 바인딩 ===
        auto parts = Split(key, '/');
        if (parts.size() >= 3) {
            std::string compName = parts[1];
            std::string propPath;
            for (size_t i = 2; i < parts.size(); i++) {
                if (i > 2) propPath += "/";
                propPath += parts[i];
            }
            ParseGeometryBinding(compName, propPath, value);
        }
    }
    else if (key.substr(0, 3) == "Ph/" || key.find("Physics") != std::string::npos) {
        // === 물리 설정 ===
        ParsePhysicsSetting(key, value);
    }

    return 0;
}

// ============================================================
// 물질 속성 파싱
// ============================================================
void TopasParameterParser::ParseMaterialProperty(
    const std::string& matName,
    const std::string& propPath,
    const std::string& /*typePrefix*/,
    const std::string& valueStr)
{
    ParsedMaterial& mat = GetOrCreateMaterial(matName);
    std::string prop = ToLower(propPath);

    // --- EnableOpticalProperties ---
    if (prop == "enableopticalproperties") {
        std::string v = ToLower(Trim(valueStr));
        v.erase(std::remove(v.begin(), v.end(), '"'), v.end());
        mat.props.isOpticalEnabled = (v == "true" || v == "1") ? 1 : 0;
        return;
    }

    // --- 벡터 속성: Energies 와 Values 쌍 ---
    // RefractiveIndex/Energies, RefractiveIndex/Values
    // RIndex/Energies, RIndex/Values (별칭)
    // AbsLength/Energies, AbsLength/Values 등

    // 에너지 벡터 저장 (나중에 Values와 매칭)
    if (prop.find("/energies") != std::string::npos) {
        // "dv:Ma/Name/RefractiveIndex/Energies = 3 2.0 2.5 3.0 eV"
        auto tokens = Split(valueStr, ' ');
        if (tokens.size() < 2) return;

        int count = std::stoi(tokens[0]);
        std::vector<float> energies;
        std::string unit = "eV";

        for (int i = 1; i <= count && i < (int)tokens.size(); i++) {
            try { energies.push_back(std::stof(tokens[i])); }
            catch (...) { break; }
        }

        // 마지막 토큰이 단위일 수 있음
        if (tokens.size() > (size_t)(count + 1)) {
            unit = tokens[count + 1];
        }

        // 에너지 단위 변환 → eV
        for (auto& e : energies) {
            e = ConvertEnergyUnit(e, unit);
        }

        // 속성 경로에서 /Energies 제거하여 키 생성
        std::string baseKey = matName + "/" + propPath.substr(0, propPath.find("/Energies"));
        if (baseKey.back() == '/') baseKey.pop_back();

        m_pendingEnergies[baseKey] = {energies, "eV"};
        return;
    }

    // Values 벡터 처리
    if (prop.find("/values") != std::string::npos) {
        auto tokens = Split(valueStr, ' ');
        if (tokens.size() < 2) return;

        int count = std::stoi(tokens[0]);
        std::vector<float> values;
        std::string unit = "";

        for (int i = 1; i <= count && i < (int)tokens.size(); i++) {
            try { values.push_back(std::stof(tokens[i])); }
            catch (...) { break; }
        }

        if (tokens.size() > (size_t)(count + 1)) {
            unit = tokens[count + 1];
        }

        // 매칭할 에너지 벡터 찾기
        std::string baseKey = matName + "/" + propPath.substr(0, propPath.find("/Values"));
        if (baseKey.back() == '/') baseKey.pop_back();

        std::string baseProp = propPath.substr(0, propPath.find("/Values"));
        std::string lowerBase = ToLower(baseProp);

        auto it = m_pendingEnergies.find(baseKey);
        std::vector<float> energies;
        if (it != m_pendingEnergies.end()) {
            energies = it->second.energies;
        }

        // 길이 단위 변환 (AbsLength 등)
        if (lowerBase.find("length") != std::string::npos ||
            lowerBase.find("abslength") != std::string::npos ||
            lowerBase.find("wlsabslength") != std::string::npos) {
            if (!unit.empty()) {
                for (auto& v : values) v = ConvertLengthUnit(v, unit);
            }
        }

        // 속성 테이블에 매핑
        MOPPropertyTable* targetTable = nullptr;
        if (lowerBase == "refractiveindex" || lowerBase == "rindex")
            targetTable = &mat.props.refractiveIndex;
        else if (lowerBase == "abslength" || lowerBase == "absorptionlength")
            targetTable = &mat.props.absorptionLength;
        else if (lowerBase == "fastcomponent")
            targetTable = &mat.props.fastComponent;
        else if (lowerBase == "slowcomponent")
            targetTable = &mat.props.slowComponent;
        else if (lowerBase == "wlsabslength")
            targetTable = &mat.props.wlsAbsLength;
        else if (lowerBase == "wlscomponent")
            targetTable = &mat.props.wlsComponent;
        else if (lowerBase == "miehg")
            targetTable = &mat.props.mieScattering;

        if (targetTable) {
            FillPropertyTable(*targetTable, energies, values);
        }
        return;
    }

    // --- 스칼라 속성 ---
    auto tokens = Split(valueStr, ' ');
    float val = 0.0f;
    std::string unit = "";

    if (!tokens.empty()) {
        try { val = std::stof(tokens[0]); } catch (...) { return; }
    }
    if (tokens.size() > 1) unit = tokens[1];

    if (prop == "scintillationyield")
        mat.props.scintillationYield = val;
    else if (prop == "resolutionscale")
        mat.props.resolutionScale = val;
    else if (prop == "fasttimeconstant")
        mat.props.fastTimeConstant = ConvertTimeUnit(val, unit.empty() ? "ns" : unit);
    else if (prop == "slowtimeconstant")
        mat.props.slowTimeConstant = ConvertTimeUnit(val, unit.empty() ? "ns" : unit);
    else if (prop == "yieldratio")
        mat.props.yieldRatio = val;
    else if (prop == "birksconstant")
        mat.props.birksConstant = val;  // mm/MeV
    else if (prop == "wlstimeconstant")
        mat.props.wlsTimeConstant = ConvertTimeUnit(val, unit.empty() ? "ns" : unit);
    else if (prop == "miehgforward")
        mat.props.miehgForward = val;
    else if (prop == "miehgbackward")
        mat.props.miehgBackward = val;
    else if (prop == "miehgforwardratio")
        mat.props.miehgForwardRatio = val;
}

// ============================================================
// 표면 속성 파싱
// ============================================================
void TopasParameterParser::ParseSurfaceProperty(
    const std::string& surfName,
    const std::string& propName,
    const std::string& /*typePrefix*/,
    const std::string& valueStr)
{
    ParsedSurface& surf = GetOrCreateSurface(surfName);
    std::string prop = ToLower(propName);
    std::string val = ParseString(valueStr);
    std::string lval = ToLower(val);

    if (prop == "type") {
        if (lval == "dielectric_metal")
            surf.props.type = MOP_SURFACE_DIELECTRIC_METAL;
        else
            surf.props.type = MOP_SURFACE_DIELECTRIC_DIELECTRIC;
    }
    else if (prop == "finish") {
        if (lval == "polished")                surf.props.finish = MOP_FINISH_POLISHED;
        else if (lval == "polishedfrontpainted") surf.props.finish = MOP_FINISH_POLISHED_FRONT_PAINT;
        else if (lval == "polishedbackpainted")  surf.props.finish = MOP_FINISH_POLISHED_BACK_PAINT;
        else if (lval == "ground")               surf.props.finish = MOP_FINISH_GROUND;
        else if (lval == "groundfrontpainted")   surf.props.finish = MOP_FINISH_GROUND_FRONT_PAINT;
        else if (lval == "groundbackpainted")    surf.props.finish = MOP_FINISH_GROUND_BACK_PAINT;
    }
    else if (prop == "model") {
        if (lval == "unified") surf.props.model = MOP_MODEL_UNIFIED;
        else                   surf.props.model = MOP_MODEL_GLISUR;
    }
    else if (prop == "sigmaalpha") {
        try { surf.props.sigmaAlpha = std::stof(val); } catch (...) {}
    }
    else if (prop == "forcereflectivity") {
        // TOPAS extension: b:Su/XXX/ForceReflectivity = "True"
        // Strict reflectivity override mode — surface.reflectivity 값을
        // 절대 반사 확률로 사용 (Geant4 의 Fresnel-gate 해석 우회).
        // 의도된 R 을 정확히 강제하고 싶을 때 (예: 거울 50 % 반사 정확히).
        if (lval == "true" || lval == "1" || lval == "yes")
            surf.props.forceReflectivity = 1;
        else
            surf.props.forceReflectivity = 0;
    }
    else if (prop.find("energies") != std::string::npos) {
        // 표면 에너지 벡터
        // 두 가지 형태 지원:
        //   dv:Su/Name/Energies = ...         (flat)
        //   dv:Su/Name/Reflectivity/Energies = ...  (nested)
        auto tokens = Split(valueStr, ' ');
        if (tokens.size() < 2) return;
        int count = std::stoi(tokens[0]);
        std::vector<float> energies;
        std::string unit = "eV";
        for (int i = 1; i <= count && i < (int)tokens.size(); i++) {
            try { energies.push_back(std::stof(tokens[i])); } catch (...) {}
        }
        if (tokens.size() > (size_t)(count + 1)) unit = tokens[count + 1];
        for (auto& e : energies) e = ConvertEnergyUnit(e, unit);

        // 부모 속성 이름 추출 (있으면): Reflectivity/Energies → Reflectivity
        std::string parentProp = "";
        auto slashPos = prop.find('/');
        if (slashPos != std::string::npos) {
            parentProp = prop.substr(0, slashPos);
        }

        // 에너지 키: 속성별 또는 범용
        std::string energyKey = "Su/" + surfName;
        if (!parentProp.empty()) energyKey += "/" + parentProp;

        m_pendingEnergies[energyKey] = {energies, "eV"};
    }
    else {
        // 표면 벡터 속성 (Reflectivity, Efficiency, Transmittance 등)
        // 두 가지 형태 지원:
        //   uv:Su/Name/Reflectivity = N v1 v2 ...         (flat)
        //   uv:Su/Name/Reflectivity/Values = N v1 v2 ...  (nested)
        auto tokens = Split(valueStr, ' ');
        if (tokens.size() < 2) return;

        int count = 0;
        try { count = std::stoi(tokens[0]); } catch (...) { return; }

        std::vector<float> values;
        for (int i = 1; i <= count && i < (int)tokens.size(); i++) {
            try { values.push_back(std::stof(tokens[i])); } catch (...) {}
        }

        // 실제 속성 이름 추출 (nested 경우 /Values 제거)
        std::string actualProp = prop;
        auto valuesPos = prop.find("/values");
        if (valuesPos != std::string::npos) {
            actualProp = prop.substr(0, valuesPos);
        }

        // 매칭할 에너지 벡터 찾기 (속성별 키 우선, 범용 키 폴백)
        std::string specificKey = "Su/" + surfName + "/" + actualProp;
        std::string genericKey = "Su/" + surfName;
        std::vector<float> energies;

        auto it = m_pendingEnergies.find(specificKey);
        if (it != m_pendingEnergies.end()) {
            energies = it->second.energies;
        } else {
            it = m_pendingEnergies.find(genericKey);
            if (it != m_pendingEnergies.end()) energies = it->second.energies;
        }

        MOPPropertyTable* target = nullptr;
        if (actualProp == "reflectivity")           target = &surf.props.reflectivity;
        else if (actualProp == "efficiency")        target = &surf.props.efficiency;
        else if (actualProp == "transmittance")     target = &surf.props.transmittance;
        else if (actualProp == "specularlobe" || actualProp == "specularlobeconstant")
                                                    target = &surf.props.specularLobe;
        else if (actualProp == "specularspike" || actualProp == "specularspikeconstant")
                                                    target = &surf.props.specularSpike;
        else if (actualProp == "backscatter" || actualProp == "backscatterconstant")
                                                    target = &surf.props.backScatter;

        if (target) FillPropertyTable(*target, energies, values);
    }
}

// ============================================================
// 지오메트리 광학 바인딩 파싱
// ============================================================
void TopasParameterParser::ParseGeometryBinding(
    const std::string& compName,
    const std::string& propPath,
    const std::string& valueStr)
{
    std::string surfaceName = ParseString(valueStr);
    if (surfaceName.empty()) return;

    // 표면 이름 정규화: 등록된 표면과 대소문자 무시 매칭
    // 예: "surfaceMirror" → "SurfaceMirror" (정의된 이름으로 교정)
    std::string normalizedName = surfaceName;
    std::string lowerSurf = ToLower(surfaceName);
    for (const auto& pair : m_surfaceNameMap) {
        if (ToLower(pair.first) == lowerSurf) {
            normalizedName = pair.first;  // 등록된 정확한 이름 사용
            break;
        }
    }

    // s:Ge/Comp1/OpticalBehaviorTo/Comp2 = "SurfaceName"
    if (propPath.find("OpticalBehaviorTo/") == 0) {
        std::string comp2 = propPath.substr(std::string("OpticalBehaviorTo/").length());
        ParsedSurfaceBinding binding;
        binding.volumeName1 = compName;
        binding.volumeName2 = comp2;
        binding.surfaceName = normalizedName;
        binding.isSkinSurface = false;
        m_bindings.push_back(binding);
    }
    // s:Ge/Comp1/OpticalBehavior = "SurfaceName"
    else if (propPath == "OpticalBehavior") {
        ParsedSurfaceBinding binding;
        binding.volumeName1 = compName;
        binding.volumeName2 = "";
        binding.surfaceName = normalizedName;
        binding.isSkinSurface = true;
        m_bindings.push_back(binding);
    }
}

// ============================================================
// 물리 설정 파싱
// ============================================================
void TopasParameterParser::ParsePhysicsSetting(const std::string& key, const std::string& value) {
    std::string lkey = ToLower(key);
    std::string lval = ToLower(Trim(value));
    lval.erase(std::remove(lval.begin(), lval.end(), '"'), lval.end());

    // ph/Default/Modules 에서 g4optical 포함 여부
    if (lkey.find("modules") != std::string::npos) {
        if (lval.find("g4optical") != std::string::npos) {
            // 광학 물리가 활성화됨 - 이미 기본값이 true
        }
    }

    // 체렌코프 설정
    if (lkey.find("cerenkov") != std::string::npos) {
        if (lkey.find("maxnumphotonsperstep") != std::string::npos ||
            lkey.find("maxphotons") != std::string::npos) {
            try { m_cerenkovMaxPhotons = std::stoi(lval); } catch (...) {}
        }
        if (lkey.find("maxbetachange") != std::string::npos) {
            try { m_cerenkovMaxBetaChange = std::stof(lval); } catch (...) {}
        }
    }

    // 2026-05-18: AllowMT — b:Ph/Default/GPUOptical/AllowMT = "True"
    if (lkey.find("gpuoptical/allowmt") != std::string::npos) {
        std::string v = lval;
        std::transform(v.begin(), v.end(), v.begin(), ::tolower);
        m_gpuOpticalAllowMT = (v == "true" || v == "\"true\"" || v == "1");
    }
}

// ============================================================
// 헬퍼 함수들
// ============================================================
ParsedMaterial& TopasParameterParser::GetOrCreateMaterial(const std::string& name) {
    auto it = m_materialNameMap.find(name);
    if (it != m_materialNameMap.end()) {
        return m_materials[it->second];
    }
    ParsedMaterial mat;
    mat.name = name;
    mat.props.materialId = (uint32_t)m_materials.size();
    mat.props.isOpticalEnabled = 1;  // 속성 정의 시 기본 활성화
    m_materialNameMap[name] = (int)m_materials.size();
    m_materials.push_back(mat);
    return m_materials.back();
}

ParsedSurface& TopasParameterParser::GetOrCreateSurface(const std::string& name) {
    auto it = m_surfaceNameMap.find(name);
    if (it != m_surfaceNameMap.end()) {
        return m_surfaces[it->second];
    }
    ParsedSurface surf;
    surf.name = name;
    surf.props.surfaceId = (uint32_t)m_surfaces.size();
    m_surfaceNameMap[name] = (int)m_surfaces.size();
    m_surfaces.push_back(surf);
    return m_surfaces.back();
}

void TopasParameterParser::FillPropertyTable(
    MOPPropertyTable& table,
    const std::vector<float>& energies,
    const std::vector<float>& values)
{
    uint32_t count = (uint32_t)std::min(values.size(), (size_t)MOP_MAX_PROPERTY_ENTRIES);
    table.count = count;

    for (uint32_t i = 0; i < count; i++) {
        table.values[i] = values[i];
        if (i < energies.size()) {
            table.energies[i] = energies[i];
        }
    }
}

int TopasParameterParser::FindMaterialByName(const std::string& name) const {
    auto it = m_materialNameMap.find(name);
    return (it != m_materialNameMap.end()) ? it->second : -1;
}

int TopasParameterParser::FindSurfaceByName(const std::string& name) const {
    // 1차: 정확한 이름 매칭
    auto it = m_surfaceNameMap.find(name);
    if (it != m_surfaceNameMap.end()) return it->second;

    // 2차: 대소문자 무시 매칭 (TOPAS 스크립트에서 흔한 불일치 처리)
    std::string lowerName = ToLower(name);
    for (const auto& pair : m_surfaceNameMap) {
        if (ToLower(pair.first) == lowerName) return pair.second;
    }
    return -1;
}

std::string TopasParameterParser::ParseString(const std::string& str) {
    std::string s = Trim(str);
    if (!s.empty() && s.front() == '"') s = s.substr(1);
    if (!s.empty() && s.back() == '"')  s.pop_back();
    return s;
}

std::vector<float> TopasParameterParser::ParseFloatVector(const std::string& str, int /*expectedCount*/) {
    auto tokens = Split(str, ' ');
    std::vector<float> result;
    for (const auto& t : tokens) {
        try { result.push_back(std::stof(t)); }
        catch (...) { break; }
    }
    return result;
}

float TopasParameterParser::ParseFloat(const std::string& str) {
    try { return std::stof(Trim(str)); }
    catch (...) { return 0.0f; }
}
