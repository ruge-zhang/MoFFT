cmake_minimum_required(VERSION 3.24)

# The published source kit contains frozen SME2 kernels; it intentionally does
# not invoke the developer generator, Python, or any tuning/calibration tools.
if(NOT DEFINED CMAKE_OSX_DEPLOYMENT_TARGET OR
   CMAKE_OSX_DEPLOYMENT_TARGET STREQUAL "")
  set(CMAKE_OSX_DEPLOYMENT_TARGET "26.4" CACHE STRING
      "Minimum macOS version supported by the bundled SME2 kernels")
endif()

project(MoFFT VERSION 0.4.0 LANGUAGES C ASM)

include(cmake/MoFFTReleaseVersion.cmake)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CheckCCompilerFlag)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)

if(NOT APPLE)
  message(FATAL_ERROR "This preview source kit targets macOS on Apple arm64 with SME2.")
endif()

set(MOFFT_TARGET_PROCESSOR "${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_OSX_ARCHITECTURES)
  if(NOT CMAKE_OSX_ARCHITECTURES MATCHES "^(arm64|arm64e)(;arm64|;arm64e)*$")
    message(FATAL_ERROR "The bundled kernels require an Apple arm64 target.")
  endif()
  set(MOFFT_TARGET_PROCESSOR "${CMAKE_OSX_ARCHITECTURES}")
elseif(NOT MOFFT_TARGET_PROCESSOR MATCHES "^(arm64|aarch64|arm64e)$")
  message(FATAL_ERROR "The bundled kernels require an Apple arm64 target.")
endif()

if(CMAKE_OSX_DEPLOYMENT_TARGET VERSION_LESS "26.4")
  message(FATAL_ERROR "The bundled preview kernels require macOS 26.4 or newer.")
endif()
if(NOT CMAKE_C_COMPILER_ID MATCHES "^(AppleClang|Clang)$")
  message(FATAL_ERROR "Use Apple Clang or a compatible Clang with SME2 support.")
endif()

check_c_compiler_flag("-march=armv9.2-a+sme+sme2+sme-f64f64"
                      MOFFT_HAS_SME2_COMPILE_FLAG)
check_c_compiler_flag("-mno-implicit-sme" MOFFT_HAS_NO_IMPLICIT_SME_FLAG)
if(NOT MOFFT_HAS_SME2_COMPILE_FLAG OR NOT MOFFT_HAS_NO_IMPLICIT_SME_FLAG)
  message(FATAL_ERROR "The selected Clang does not accept the SME2 kernel flags required by this preview.")
endif()

set(MOFFT_GENERATED_DIR "${CMAKE_CURRENT_SOURCE_DIR}/generated")
set(MOFFT_KERNEL_SOURCES
  "${MOFFT_GENERATED_DIR}/mofft_kernels_fp32.c"
  "${MOFFT_GENERATED_DIR}/mofft_kernels_fp64.c"
  "${MOFFT_GENERATED_DIR}/mofft_kernel_registry.c")
set(MOFFT_PORTFOLIO_SOURCES
  "${MOFFT_GENERATED_DIR}/mofft_kernel_portfolio.c")
foreach(bank RANGE 1 3)
  list(APPEND MOFFT_KERNEL_SOURCES
    "${MOFFT_GENERATED_DIR}/bank${bank}/mofft_kernels_fp32.c"
    "${MOFFT_GENERATED_DIR}/bank${bank}/mofft_kernels_fp64.c"
    "${MOFFT_GENERATED_DIR}/bank${bank}/mofft_kernel_registry.c")
endforeach()

set(MOFFT_ID_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/mofft.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/mofft.c"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/streaming_entry.S"
  "${MOFFT_GENERATED_DIR}/mofft_generated_kernels.h"
  "${MOFFT_GENERATED_DIR}/mofft_generated_wisdom.h"
  "${MOFFT_GENERATED_DIR}/mofft_generated_wisdom.c"
  "${MOFFT_GENERATED_DIR}/mofft_kernel_portfolio.h"
  "${MOFFT_GENERATED_DIR}/mofft_kernel_portfolio.c"
  "${MOFFT_GENERATED_DIR}/mofft_kernel_registry.c"
  "${MOFFT_GENERATED_DIR}/mofft_kernels_fp32.c"
  "${MOFFT_GENERATED_DIR}/mofft_kernels_fp64.c"
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/MoFFTConfig.cmake.in"
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/mofft-build-info.json.in")
foreach(bank RANGE 1 3)
  list(APPEND MOFFT_ID_SOURCES
    "${MOFFT_GENERATED_DIR}/bank${bank}/mofft_generated_kernels.h"
    "${MOFFT_GENERATED_DIR}/bank${bank}/mofft_kernel_registry.c"
    "${MOFFT_GENERATED_DIR}/bank${bank}/mofft_kernels_fp32.c"
    "${MOFFT_GENERATED_DIR}/bank${bank}/mofft_kernels_fp64.c")
endforeach()
list(SORT MOFFT_ID_SOURCES)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${MOFFT_ID_SOURCES})

# Recipe compatibility is tied to this exact frozen code set and the effective
# compiler/OS/architecture settings, not to the prebuilt SDK's build identity.
set(MOFFT_BINARY_SIGNATURE "MoFFT-source-kit-recipe-v1|${PROJECT_VERSION}")
foreach(source IN LISTS MOFFT_ID_SOURCES)
  if(NOT EXISTS "${source}")
    message(FATAL_ERROR "Source kit file is missing: ${source}")
  endif()
  file(SHA256 "${source}" source_digest)
  file(RELATIVE_PATH source_name "${CMAKE_CURRENT_SOURCE_DIR}" "${source}")
  string(APPEND MOFFT_BINARY_SIGNATURE "|${source_name}:${source_digest}")
endforeach()
string(TOUPPER "${CMAKE_BUILD_TYPE}" MOFFT_BUILD_TYPE_UPPER)
string(APPEND MOFFT_BINARY_SIGNATURE
  "|${CMAKE_SYSTEM_NAME}|${MOFFT_TARGET_PROCESSOR}"
  "|${CMAKE_C_COMPILER_ID}|${CMAKE_C_COMPILER_VERSION}|${CMAKE_C_COMPILER_TARGET}"
  "|${CMAKE_ASM_COMPILER_ID}|${CMAKE_ASM_COMPILER_VERSION}"
  "|${CMAKE_OSX_DEPLOYMENT_TARGET}|${CMAKE_OSX_SYSROOT}|${CMAKE_OSX_ARCHITECTURES}"
  "|${CMAKE_SYSTEM_VERSION}|${CMAKE_BUILD_TYPE}"
  "|${CMAKE_C_FLAGS}|${CMAKE_C_FLAGS_${MOFFT_BUILD_TYPE_UPPER}}"
  "|${CMAKE_ASM_FLAGS}|${CMAKE_ASM_FLAGS_${MOFFT_BUILD_TYPE_UPPER}}"
  "|-O3|-fomit-frame-pointer|-funroll-loops|-g0"
  "|-march=armv9.2-a+nosve|-march=armv9.2-a+sme+sme2+sme-f64f64"
  "|-mno-implicit-sme|MOFFT_FUSE_FIRST_TRANSPOSE=1"
  "|MOFFT_BROADCAST_REPEATED_TWIDDLES=1|four-kernel-banks")
string(SHA256 MOFFT_BINARY_ID "${MOFFT_BINARY_SIGNATURE}")

add_library(mofft STATIC
  "${CMAKE_CURRENT_SOURCE_DIR}/src/mofft.c"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/streaming_entry.S"
  "${MOFFT_GENERATED_DIR}/mofft_generated_wisdom.c"
  ${MOFFT_KERNEL_SOURCES}
  ${MOFFT_PORTFOLIO_SOURCES})
add_library(MoFFT::mofft ALIAS mofft)
set_target_properties(mofft PROPERTIES
  EXPORT_NAME mofft
  OUTPUT_NAME mofft
  POSITION_INDEPENDENT_CODE ON)

target_compile_definitions(mofft PRIVATE
  "MOFFT_BINARY_ID=\"${MOFFT_BINARY_ID}\""
  MOFFT_HAVE_KERNEL_PORTFOLIO=1
  MOFFT_FUSE_FIRST_TRANSPOSE=1
  MOFFT_BROADCAST_REPEATED_TWIDDLES=1)
target_compile_features(mofft PUBLIC c_std_11)
target_include_directories(mofft
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
  PRIVATE "${MOFFT_GENERATED_DIR}")
target_compile_options(mofft PRIVATE
  -O3 -fomit-frame-pointer -funroll-loops -march=armv9.2-a+nosve -g0
  "-ffile-prefix-map=${CMAKE_CURRENT_SOURCE_DIR}=."
  "-fdebug-prefix-map=${CMAKE_CURRENT_SOURCE_DIR}=."
  "-fmacro-prefix-map=${CMAKE_CURRENT_SOURCE_DIR}=."
  "-ffile-prefix-map=${CMAKE_CURRENT_BINARY_DIR}=.")
target_link_libraries(mofft PUBLIC m)

set_source_files_properties(${MOFFT_KERNEL_SOURCES}
  PROPERTIES COMPILE_OPTIONS "-march=armv9.2-a+sme+sme2+sme-f64f64;-mno-implicit-sme")
set_source_files_properties(${MOFFT_PORTFOLIO_SOURCES}
  PROPERTIES COMPILE_OPTIONS "-march=armv9.2-a+nosve")

set(MOFFT_PLAN_WISDOM_CASES)
file(STRINGS "${MOFFT_GENERATED_DIR}/mofft_generated_wisdom.c"
     MOFFT_PLAN_WISDOM_CASES REGEX "if \\(length ==")
list(LENGTH MOFFT_PLAN_WISDOM_CASES MOFFT_PLAN_WISDOM_ENTRY_COUNT)
configure_file(cmake/mofft-build-info.json.in
               "${CMAKE_CURRENT_BINARY_DIR}/mofft-build-info.json" @ONLY)

configure_package_config_file(
  cmake/MoFFTConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/MoFFTConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/MoFFT")
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/MoFFTConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY SameMajorVersion)

install(TARGETS mofft EXPORT MoFFTTargets
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(FILES include/mofft.h DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(EXPORT MoFFTTargets FILE MoFFTTargets.cmake NAMESPACE MoFFT::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/MoFFT")
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/MoFFTConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/MoFFTConfigVersion.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/MoFFT")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/mofft-build-info.json"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/mofft")
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/MoFFT")

include(CTest)
if(BUILD_TESTING)
  add_executable(mofft-source-smoke tests/source_smoke.c)
  target_link_libraries(mofft-source-smoke PRIVATE MoFFT::mofft)
  target_compile_options(mofft-source-smoke PRIVATE -O2)
  add_test(NAME source-smoke COMMAND mofft-source-smoke)
endif()
