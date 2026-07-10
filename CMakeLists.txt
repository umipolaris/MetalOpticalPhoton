##############################################################################
# MetalOpticalPhoton - GPU 가속 광학 광자 전파 라이브러리
# Apple Metal 기반, TOPAS/Geant4 통합
#
# 빌드 방법:
#   mkdir build && cd build
#   cmake .. -DTOPAS_DIR=/path/to/topas -DGeant4_DIR=/path/to/geant4
#   make -j$(sysctl -n hw.ncpu)
#
# TOPAS Extension으로 사용:
#   cmake /path/to/topas -DTOPAS_EXTENSIONS_DIR=/path/to/MetalOpticalPhoton/topas_extension
##############################################################################

cmake_minimum_required(VERSION 3.20)
project(MetalOpticalPhoton
    VERSION 1.0.0
    DESCRIPTION "GPU-accelerated optical photon propagation using Apple Metal"
    LANGUAGES CXX OBJCXX
)

# ============================================================
# 빌드 옵션
# ============================================================
option(MOP_BUILD_STANDALONE  "Build standalone library (without TOPAS)" ON)
option(MOP_BUILD_TOPAS_EXT   "Build TOPAS extension"                   OFF)
option(MOP_BUILD_TESTS       "Build unit tests"                        OFF)
option(MOP_ENABLE_RAYTRACING "Enable Metal Ray Tracing API (M3+)"     OFF)

# C++ 표준
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 17)

# macOS 최소 버전 (Metal 3.0 지원)
set(CMAKE_OSX_DEPLOYMENT_TARGET "13.0" CACHE STRING "Minimum macOS version")

# ============================================================
# Apple 프레임워크
# ============================================================
find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(METALPS_FRAMEWORK MetalPerformanceShaders REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)

# ============================================================
# Metal 셰이더 컴파일
# ============================================================
set(SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
set(SHADER_SOURCES
    ${SHADER_DIR}/PhotonGeneration.metal
    ${SHADER_DIR}/OpticalPhotonKernel.metal
    ${SHADER_DIR}/DDAScoring.metal
)
# Note: Common.metal 제거됨 — Common.h가 유일한 정의 소스 (ODR 위반 방지)

# Metal 셰이더 → .air → .metallib 변환
set(METALLIB_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/MetalOpticalPhoton.metallib")
set(AIR_OUTPUTS "")

foreach(SHADER_SRC ${SHADER_SOURCES})
    get_filename_component(SHADER_NAME ${SHADER_SRC} NAME_WE)
    set(AIR_FILE "${CMAKE_CURRENT_BINARY_DIR}/${SHADER_NAME}.air")

    add_custom_command(
        OUTPUT ${AIR_FILE}
        COMMAND xcrun -sdk macosx metal
            -c ${SHADER_SRC}
            -o ${AIR_FILE}
            -std=metal3.0
            -O2
            -ffast-math
        DEPENDS ${SHADER_SRC} ${SHADER_DIR}/Common.h
        COMMENT "Compiling Metal shader: ${SHADER_NAME}.metal"
    )

    list(APPEND AIR_OUTPUTS ${AIR_FILE})
endforeach()

add_custom_command(
    OUTPUT ${METALLIB_OUTPUT}
    COMMAND xcrun -sdk macosx metallib
        ${AIR_OUTPUTS}
        -o ${METALLIB_OUTPUT}
    DEPENDS ${AIR_OUTPUTS}
    COMMENT "Linking Metal shader library"
)

add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/default.metallib"
    COMMAND ${CMAKE_COMMAND} -E copy ${METALLIB_OUTPUT} "${CMAKE_CURRENT_BINARY_DIR}/default.metallib"
    DEPENDS ${METALLIB_OUTPUT}
    COMMENT "Copying MetalOpticalPhoton.metallib → default.metallib"
)
add_custom_target(MetalShaders ALL DEPENDS ${METALLIB_OUTPUT} "${CMAKE_CURRENT_BINARY_DIR}/default.metallib")

# ============================================================
# 1. 독립 라이브러리 (libMetalOpticalPhoton.dylib)
# ============================================================
if(MOP_BUILD_STANDALONE)
    set(MOP_SOURCES
        src/MetalOpticalEngine.mm
        topas_extension/TopasParameterParser.cc
    )

    set(MOP_HEADERS
        include/MOPTypes.h
        include/MetalOpticalEngine.h
        topas_extension/TopasParameterParser.hh
    )

    # 공유 라이브러리
    add_library(MetalOpticalPhoton SHARED ${MOP_SOURCES})

    target_include_directories(MetalOpticalPhoton
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/topas_extension
    )

    target_link_libraries(MetalOpticalPhoton
        PRIVATE
            ${METAL_FRAMEWORK}
            ${METALPS_FRAMEWORK}
            ${FOUNDATION_FRAMEWORK}
    )

    # Objective-C ARC 활성화
    target_compile_options(MetalOpticalPhoton PRIVATE
        $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>
    )

    # Metal 라이브러리를 리소스로 포함
    add_dependencies(MetalOpticalPhoton MetalShaders)
    target_compile_definitions(MetalOpticalPhoton PRIVATE
        MOP_METALLIB_PATH="${METALLIB_OUTPUT}"
    )

    # 정적 라이브러리도 빌드
    add_library(MetalOpticalPhoton_static STATIC ${MOP_SOURCES})
    target_include_directories(MetalOpticalPhoton_static
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/topas_extension
    )
    target_link_libraries(MetalOpticalPhoton_static
        PRIVATE ${METAL_FRAMEWORK} ${METALPS_FRAMEWORK} ${FOUNDATION_FRAMEWORK}
    )
    target_compile_options(MetalOpticalPhoton_static PRIVATE
        $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>
    )
    add_dependencies(MetalOpticalPhoton_static MetalShaders)
    target_compile_definitions(MetalOpticalPhoton_static PRIVATE
        MOP_METALLIB_PATH="${METALLIB_OUTPUT}"
    )

    # 설치
    install(TARGETS MetalOpticalPhoton MetalOpticalPhoton_static
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
    )
    install(FILES ${MOP_HEADERS} DESTINATION include/MetalOpticalPhoton)
    install(FILES ${METALLIB_OUTPUT} DESTINATION share/MetalOpticalPhoton)
endif()

# ============================================================
# 2. TOPAS Extension 빌드
# ============================================================
if(MOP_BUILD_TOPAS_EXT)
    # Geant4 찾기
    find_package(Geant4 REQUIRED)
    include(${Geant4_USE_FILE})

    # TOPAS 찾기
    if(DEFINED TOPAS_DIR)
        set(TOPAS_INCLUDE_DIRS "${TOPAS_DIR}/include")
    endif()

    set(TOPAS_EXT_SOURCES
        topas_extension/TsGPUOpticalPhysics.cc
        topas_extension/TopasParameterParser.cc
        src/MetalOpticalEngine.mm
    )

    add_library(TsGPUOpticalPhysics SHARED ${TOPAS_EXT_SOURCES})

    target_include_directories(TsGPUOpticalPhysics
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/topas_extension
            ${Geant4_INCLUDE_DIRS}
            ${TOPAS_INCLUDE_DIRS}
    )

    target_link_libraries(TsGPUOpticalPhysics
        PRIVATE
            ${Geant4_LIBRARIES}
            ${METAL_FRAMEWORK}
            ${METALPS_FRAMEWORK}
            ${FOUNDATION_FRAMEWORK}
    )

    target_compile_options(TsGPUOpticalPhysics PRIVATE
        $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>
    )

    add_dependencies(TsGPUOpticalPhysics MetalShaders)
    target_compile_definitions(TsGPUOpticalPhysics PRIVATE
        MOP_METALLIB_PATH="${METALLIB_OUTPUT}"
    )

    install(TARGETS TsGPUOpticalPhysics LIBRARY DESTINATION lib)
endif()

# ============================================================
# 3. 테스트 빌드
# ============================================================
if(MOP_BUILD_TESTS)
    enable_testing()

    # 파라미터 파서 테스트 (Geant4 불필요)
    add_executable(test_parameter_parser
        test/test_parameter_parser.cc
        topas_extension/TopasParameterParser.cc
    )
    target_include_directories(test_parameter_parser PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/topas_extension
    )
    add_test(NAME TestParameterParser COMMAND test_parameter_parser)

    # Metal 엔진 테스트 (Geant4 불필요, Metal 필요)
    add_executable(test_metal_engine
        test/test_metal_engine.mm
        src/MetalOpticalEngine.mm
        topas_extension/TopasParameterParser.cc
    )
    target_include_directories(test_metal_engine PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/topas_extension
    )
    target_link_libraries(test_metal_engine
        ${METAL_FRAMEWORK} ${METALPS_FRAMEWORK} ${FOUNDATION_FRAMEWORK}
    )
    target_compile_options(test_metal_engine PRIVATE
        $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>
    )
    add_dependencies(test_metal_engine MetalShaders)
    target_compile_definitions(test_metal_engine PRIVATE
        MOP_METALLIB_PATH="${METALLIB_OUTPUT}"
    )
    add_test(NAME TestMetalEngine COMMAND test_metal_engine)
endif()

# ============================================================
# 정보 출력
# ============================================================
message(STATUS "============================================")
message(STATUS "MetalOpticalPhoton v${PROJECT_VERSION}")
message(STATUS "  Standalone library: ${MOP_BUILD_STANDALONE}")
message(STATUS "  TOPAS extension:    ${MOP_BUILD_TOPAS_EXT}")
message(STATUS "  Tests:              ${MOP_BUILD_TESTS}")
message(STATUS "  Metal Ray Tracing:  ${MOP_ENABLE_RAYTRACING}")
message(STATUS "  macOS target:       ${CMAKE_OSX_DEPLOYMENT_TARGET}")
message(STATUS "============================================")
