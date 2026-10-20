cmake_minimum_required(VERSION 3.24)
project(MoFFT VERSION 0.4.0 LANGUAGES C CXX ASM)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)

option(MOFFT_BUILD_TESTS "Build correctness tests" ON)
option(MOFFT_BUILD_TOOLS "Build calibration and benchmark tools" ON)
option(MOFFT_WITH_FFTW "Enable the FFTW comparison benchmark" OFF)
option(MOFFT_FUSE_FIRST_TRANSPOSE
       "Let two-stage first kernels store directly in parent layout" ON)
option(MOFFT_BROADCAST_REPEATED_TWIDDLES
       "Broadcast lane-invariant twiddles in deep plans" ON)
option(MOFFT_VALIDATE_ALL_CANDIDATES
       "Compile and execute every legal generated-kernel candidate" OFF)
option(MOFFT_BUILD_KERNEL_PORTFOLIO
       "Compile the generated Like-Terms, Even-Odd, and Vector-Reuse kernel banks"
       OFF)
set(MOFFT_PROFILE "${CMAKE_CURRENT_SOURCE_DIR}/profiles/apple-m5-bootstrap.json"
    CACHE FILEPATH "Machine profile used by the kernel compiler")
set(MOFFT_WISDOM "" CACHE FILEPATH
    "Optional mofft-plan-search JSON compiled into the runtime")
set(MOFFT_WISDOM_MACHINE "" CACHE STRING
    "Expected hw.model for cross-compiling plan wisdom (defaults to build host)")
set(MOFFT_KERNEL_WISDOM "" CACHE FILEPATH
    "Optional measured expression/VG choices used by the kernel compiler")
set(MOFFT_TEMP_TILE_ROTATION "auto" CACHE STRING
    "Other-stage temporary ZA tile rotation policy: auto, on, or off")
set_property(CACHE MOFFT_TEMP_TILE_ROTATION PROPERTY STRINGS auto on off)
if(NOT MOFFT_TEMP_TILE_ROTATION MATCHES "^(auto|on|off)$")
  message(FATAL_ERROR "MOFFT_TEMP_TILE_ROTATION must be auto, on, or off")
endif()

set(MOFFT_TARGET_PROCESSOR "${CMAKE_SYSTEM_PROCESSOR}")
# CMake's iOS toolchain leaves CMAKE_SYSTEM_PROCESSOR empty and expresses the
# target architecture through CMAKE_OSX_ARCHITECTURES instead.
if(NOT MOFFT_TARGET_PROCESSOR AND CMAKE_OSX_ARCHITECTURES)
  set(MOFFT_TARGET_PROCESSOR "${CMAKE_OSX_ARCHITECTURES}")
endif()
if(NOT APPLE OR NOT MOFFT_TARGET_PROCESSOR MATCHES "^(arm64|aarch64)$")
  message(FATAL_ERROR "MoFFT requires Apple arm64 with SME support")
endif()

find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter)
# Plan recipes are reusable only with a matching generated/runtime build.
# Hash content and public build settings, never absolute source/input paths.
file(GLOB_RECURSE MOFFT_ID_COMPILER_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/compiler/mofft_compiler/*.py")
set(MOFFT_ID_SOURCES ${MOFFT_ID_COMPILER_SOURCES}
  "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/mofft.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/mofft.c"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/streaming_entry.S"
  "${CMAKE_CURRENT_SOURCE_DIR}/tools/compile_wisdom.py")
list(SORT MOFFT_ID_SOURCES)
set(MOFFT_BINARY_SIGNATURE "MoFFT-plan-recipe-v1|${PROJECT_VERSION}")
foreach(source IN LISTS MOFFT_ID_SOURCES)
  file(SHA256 "${source}" digest)
  file(RELATIVE_PATH relative "${CMAKE_CURRENT_SOURCE_DIR}" "${source}")
  string(APPEND MOFFT_BINARY_SIGNATURE "|${relative}:${digest}")
endforeach()
set(MOFFT_ID_INPUTS "${MOFFT_PROFILE}")
if(MOFFT_KERNEL_WISDOM)
  list(APPEND MOFFT_ID_INPUTS "${MOFFT_KERNEL_WISDOM}")
endif()
if(MOFFT_WISDOM)
  list(APPEND MOFFT_ID_INPUTS "${MOFFT_WISDOM}")
endif()
foreach(input IN LISTS MOFFT_ID_INPUTS)
  file(SHA256 "${input}" digest)
  string(APPEND MOFFT_BINARY_SIGNATURE "|input:${digest}")
endforeach()
string(TOUPPER "${CMAKE_BUILD_TYPE}" MOFFT_BUILD_TYPE_UPPER)
if(CMAKE_SYSTEM_NAME STREQUAL "iOS")
  set(MOFFT_ID_SDK iphoneos)
else()
  set(MOFFT_ID_SDK macosx)
endif()
execute_process(COMMAND xcrun --sdk "${MOFFT_ID_SDK}" --show-sdk-version
  OUTPUT_VARIABLE MOFFT_ID_SDK_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
string(APPEND MOFFT_BINARY_SIGNATURE
  "|${CMAKE_SYSTEM_NAME}|${MOFFT_TARGET_PROCESSOR}"
  "|${CMAKE_C_COMPILER_ID}|${CMAKE_C_COMPILER_VERSION}|${CMAKE_C_COMPILER_TARGET}"
  "|${MOFFT_ID_SDK_VERSION}|${CMAKE_OSX_DEPLOYMENT_TARGET}|${CMAKE_BUILD_TYPE}"
  "|${CMAKE_C_FLAGS}|${CMAKE_C_FLAGS_${MOFFT_BUILD_TYPE_UPPER}}"
  "|${MOFFT_BUILD_KERNEL_PORTFOLIO}|${MOFFT_FUSE_FIRST_TRANSPOSE}"
  "|${MOFFT_BROADCAST_REPEATED_TWIDDLES}|${MOFFT_TEMP_TILE_ROTATION}")
string(SHA256 MOFFT_BINARY_ID "${MOFFT_BINARY_SIGNATURE}")
file(CONFIGURE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/mofft-build-id.txt"
  CONTENT "@MOFFT_BINARY_ID@\n" @ONLY)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  ${MOFFT_ID_SOURCES} ${MOFFT_ID_INPUTS})

set(MOFFT_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(MOFFT_GENERATED
  "${MOFFT_GENERATED_DIR}/mofft_generated_kernels.h"
  "${MOFFT_GENERATED_DIR}/mofft_kernels_fp32.c"
  "${MOFFT_GENERATED_DIR}/mofft_kernels_fp64.c"
  "${MOFFT_GENERATED_DIR}/mofft_kernel_registry.c"
  "${MOFFT_GENERATED_DIR}/manifest.json")

set(MOFFT_KERNEL_WISDOM_ARGS)
set(MOFFT_KERNEL_WISDOM_DEPENDS)
if(MOFFT_KERNEL_WISDOM)
  list(APPEND MOFFT_KERNEL_WISDOM_ARGS
       --kernel-wisdom "${MOFFT_KERNEL_WISDOM}")
  list(APPEND MOFFT_KERNEL_WISDOM_DEPENDS "${MOFFT_KERNEL_WISDOM}")
endif()

add_custom_command(
  OUTPUT ${MOFFT_GENERATED}
  COMMAND ${CMAKE_COMMAND} -E make_directory "${MOFFT_GENERATED_DIR}"
  COMMAND ${CMAKE_COMMAND} -E env
          "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/compiler"
          ${Python3_EXECUTABLE} -m mofft_compiler.cli
          --profile "${MOFFT_PROFILE}" --output "${MOFFT_GENERATED_DIR}"
          --temp-tile-rotation "${MOFFT_TEMP_TILE_ROTATION}"
          ${MOFFT_KERNEL_WISDOM_ARGS}
  DEPENDS
    "${MOFFT_PROFILE}"
    compiler/mofft_compiler/__init__.py
    compiler/mofft_compiler/backends/__init__.py
    compiler/mofft_compiler/backends/arm_sme.py
    compiler/mofft_compiler/backends/base.py
    compiler/mofft_compiler/cli.py
    compiler/mofft_compiler/emitter.py
    compiler/mofft_compiler/ir.py
    compiler/mofft_compiler/mapping.py
    compiler/mofft_compiler/model.py
    compiler/mofft_compiler/patterns.py
    compiler/mofft_compiler/portfolio.py
    compiler/mofft_compiler/profile.py
    compiler/mofft_compiler/rewrites.py
    compiler/mofft_compiler/scheduler.py
    compiler/mofft_compiler/targets/__init__.py
    compiler/mofft_compiler/targets/arm_sme.py
    compiler/mofft_compiler/targets/base.py
    ${MOFFT_KERNEL_WISDOM_DEPENDS}
  VERBATIM)

add_custom_target(mofft_generate DEPENDS ${MOFFT_GENERATED})

set(MOFFT_PORTFOLIO_GENERATED_DIR "${MOFFT_GENERATED_DIR}")
set(MOFFT_PORTFOLIO_GENERATED)
if(MOFFT_BUILD_KERNEL_PORTFOLIO)
  file(GLOB_RECURSE MOFFT_COMPILER_MODULES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/compiler/mofft_compiler/*.py")
  foreach(bank RANGE 1 3)
    list(APPEND MOFFT_PORTFOLIO_GENERATED
      "${MOFFT_PORTFOLIO_GENERATED_DIR}/bank${bank}/mofft_generated_kernels.h"
      "${MOFFT_PORTFOLIO_GENERATED_DIR}/bank${bank}/mofft_kernels_fp32.c"
      "${MOFFT_PORTFOLIO_GENERATED_DIR}/bank${bank}/mofft_kernels_fp64.c"
      "${MOFFT_PORTFOLIO_GENERATED_DIR}/bank${bank}/mofft_kernel_registry.c"
      "${MOFFT_PORTFOLIO_GENERATED_DIR}/bank${bank}/manifest.json")
  endforeach()
  list(APPEND MOFFT_PORTFOLIO_GENERATED
    "${MOFFT_PORTFOLIO_GENERATED_DIR}/mofft_kernel_portfolio.h"
    "${MOFFT_PORTFOLIO_GENERATED_DIR}/mofft_kernel_portfolio.c")
  add_custom_command(
    OUTPUT ${MOFFT_PORTFOLIO_GENERATED}
    COMMAND ${CMAKE_COMMAND} -E make_directory "${MOFFT_PORTFOLIO_GENERATED_DIR}"
    COMMAND ${CMAKE_COMMAND} -E env
            "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/compiler"
            ${Python3_EXECUTABLE} -m mofft_compiler.portfolio
            --profile "${MOFFT_PROFILE}" --output "${MOFFT_PORTFOLIO_GENERATED_DIR}"
            --temp-tile-rotation "${MOFFT_TEMP_TILE_ROTATION}"
    DEPENDS
      "${MOFFT_PROFILE}"
      ${MOFFT_COMPILER_MODULES}
    VERBATIM)
  add_custom_target(mofft_generate_kernel_portfolio
    DEPENDS ${MOFFT_PORTFOLIO_GENERATED})
endif()

set(MOFFT_WISDOM_C "${MOFFT_GENERATED_DIR}/mofft_generated_wisdom.c")
set(MOFFT_WISDOM_H "${MOFFT_GENERATED_DIR}/mofft_generated_wisdom.h")
set(MOFFT_WISDOM_ARGS)
set(MOFFT_WISDOM_DEPENDS tools/compile_wisdom.py)
if(MOFFT_WISDOM)
  list(APPEND MOFFT_WISDOM_ARGS --input "${MOFFT_WISDOM}"
       --manifest "${MOFFT_GENERATED_DIR}/manifest.json")
  if(MOFFT_WISDOM_MACHINE)
    list(APPEND MOFFT_WISDOM_ARGS --target-machine "${MOFFT_WISDOM_MACHINE}")
  endif()
  list(APPEND MOFFT_WISDOM_DEPENDS "${MOFFT_WISDOM}"
       "${MOFFT_GENERATED_DIR}/manifest.json")
endif()
add_custom_command(
  OUTPUT "${MOFFT_WISDOM_C}" "${MOFFT_WISDOM_H}"
  COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/tools/compile_wisdom.py"
          ${MOFFT_WISDOM_ARGS}
          --output-c "${MOFFT_WISDOM_C}" --output-h "${MOFFT_WISDOM_H}"
  DEPENDS ${MOFFT_WISDOM_DEPENDS}
  VERBATIM)
add_custom_target(mofft_compile_wisdom
  DEPENDS "${MOFFT_WISDOM_C}" "${MOFFT_WISDOM_H}")
add_dependencies(mofft_compile_wisdom mofft_generate)

set(MOFFT_BUILD_INFO "${MOFFT_GENERATED_DIR}/mofft-build-info.json")
set(MOFFT_BUILD_INFO_ARGS)
set(MOFFT_BUILD_INFO_DEPENDS
  tools/packaging/write_build_info.py "${MOFFT_PROFILE}"
  "${CMAKE_CURRENT_BINARY_DIR}/mofft-build-id.txt"
  "${MOFFT_GENERATED_DIR}/manifest.json")
if(CMAKE_SYSTEM_NAME STREQUAL "iOS")
  set(MOFFT_BUILD_SDK ios)
else()
  set(MOFFT_BUILD_SDK macos)
endif()
list(APPEND MOFFT_BUILD_INFO_ARGS --sdk "${MOFFT_BUILD_SDK}"
  --build-id "${MOFFT_BINARY_ID}")
if(CMAKE_OSX_DEPLOYMENT_TARGET)
  list(APPEND MOFFT_BUILD_INFO_ARGS --minimum-os "${CMAKE_OSX_DEPLOYMENT_TARGET}")
endif()
if(MOFFT_WISDOM)
  list(APPEND MOFFT_BUILD_INFO_ARGS --plan-wisdom "${MOFFT_WISDOM}")
  list(APPEND MOFFT_BUILD_INFO_DEPENDS "${MOFFT_WISDOM}")
endif()
if(MOFFT_KERNEL_WISDOM)
  list(APPEND MOFFT_BUILD_INFO_ARGS --kernel-wisdom "${MOFFT_KERNEL_WISDOM}")
  list(APPEND MOFFT_BUILD_INFO_DEPENDS "${MOFFT_KERNEL_WISDOM}")
endif()
if(MOFFT_BUILD_KERNEL_PORTFOLIO)
  set(MOFFT_KERNEL_BANK_COUNT 4)
else()
  set(MOFFT_KERNEL_BANK_COUNT 1)
endif()
add_custom_command(
  OUTPUT "${MOFFT_BUILD_INFO}"
  COMMAND ${Python3_EXECUTABLE}
          "${CMAKE_CURRENT_SOURCE_DIR}/tools/packaging/write_build_info.py"
          --profile "${MOFFT_PROFILE}" --manifest "${MOFFT_GENERATED_DIR}/manifest.json"
          --kernel-bank-count "${MOFFT_KERNEL_BANK_COUNT}"
          --output "${MOFFT_BUILD_INFO}" ${MOFFT_BUILD_INFO_ARGS}
  DEPENDS ${MOFFT_BUILD_INFO_DEPENDS}
  VERBATIM)
add_custom_target(mofft_build_info DEPENDS "${MOFFT_BUILD_INFO}")
add_dependencies(mofft_build_info mofft_generate)

add_library(mofft STATIC
  src/mofft.c
  src/streaming_entry.S
  "${MOFFT_WISDOM_C}"
  "${MOFFT_GENERATED_DIR}/mofft_kernels_fp32.c"
  "${MOFFT_GENERATED_DIR}/mofft_kernels_fp64.c"
  "${MOFFT_GENERATED_DIR}/mofft_kernel_registry.c")
target_compile_definitions(mofft PRIVATE "MOFFT_BINARY_ID=\"${MOFFT_BINARY_ID}\"")
add_dependencies(mofft mofft_generate mofft_compile_wisdom mofft_build_info)
if(MOFFT_BUILD_KERNEL_PORTFOLIO)
  set(MOFFT_PORTFOLIO_SOURCES)
  foreach(bank RANGE 1 3)
    list(APPEND MOFFT_PORTFOLIO_SOURCES
      "${MOFFT_PORTFOLIO_GENERATED_DIR}/bank${bank}/mofft_kernels_fp32.c"
      "${MOFFT_PORTFOLIO_GENERATED_DIR}/bank${bank}/mofft_kernels_fp64.c"
      "${MOFFT_PORTFOLIO_GENERATED_DIR}/bank${bank}/mofft_kernel_registry.c")
  endforeach()
  target_sources(mofft PRIVATE
    "${MOFFT_PORTFOLIO_GENERATED_DIR}/mofft_kernel_portfolio.c"
    ${MOFFT_PORTFOLIO_SOURCES})
  add_dependencies(mofft mofft_generate_kernel_portfolio)
  target_compile_definitions(mofft PRIVATE MOFFT_HAVE_KERNEL_PORTFOLIO=1)
  set_source_files_properties(${MOFFT_PORTFOLIO_SOURCES}
    PROPERTIES COMPILE_OPTIONS "-march=armv9.2-a+sme+sme2+sme-f64f64;-mno-implicit-sme")
  set_source_files_properties(
    "${MOFFT_PORTFOLIO_GENERATED_DIR}/mofft_kernel_portfolio.c"
    PROPERTIES COMPILE_OPTIONS "-march=armv9.2-a+nosve")
endif()
target_include_directories(mofft
  PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
         "$<INSTALL_INTERFACE:include>"
  PRIVATE "${MOFFT_GENERATED_DIR}")
target_compile_options(mofft PRIVATE
  -O3 -fomit-frame-pointer -funroll-loops -march=armv9.2-a+nosve -g0
  "-ffile-prefix-map=${CMAKE_CURRENT_SOURCE_DIR}=."
  "-fdebug-prefix-map=${CMAKE_CURRENT_SOURCE_DIR}=."
  "-fmacro-prefix-map=${CMAKE_CURRENT_SOURCE_DIR}=."
  "-ffile-prefix-map=${CMAKE_CURRENT_BINARY_DIR}=.")
# SME/SVE is needed only by the generated kernels (which use SME intrinsics).
# The runtime src/mofft.c is plain C; compiling it without +sme avoids SVE
# register saves (`addvl`) in its non-streaming functions, which are illegal
# on Apple Silicon outside streaming mode. The streaming functions in mofft.c
# enable SME per-function via `__attribute__((target("sme")))`.
set_source_files_properties(
  ${MOFFT_GENERATED_DIR}/mofft_kernels_fp32.c
  ${MOFFT_GENERATED_DIR}/mofft_kernels_fp64.c
  ${MOFFT_GENERATED_DIR}/mofft_kernel_registry.c
  PROPERTIES COMPILE_OPTIONS "-march=armv9.2-a+sme+sme2+sme-f64f64;-mno-implicit-sme")
if(MOFFT_FUSE_FIRST_TRANSPOSE)
  target_compile_definitions(mofft PRIVATE MOFFT_FUSE_FIRST_TRANSPOSE=1)
endif()
if(MOFFT_BROADCAST_REPEATED_TWIDDLES)
  target_compile_definitions(mofft PRIVATE MOFFT_BROADCAST_REPEATED_TWIDDLES=1)
endif()
target_link_libraries(mofft PUBLIC m)

if(MOFFT_BUILD_TESTS)
  enable_testing()
  add_executable(mofft_correctness tests/test_fft.c)
  target_link_libraries(mofft_correctness PRIVATE mofft)
  target_compile_options(mofft_correctness PRIVATE -O2)
  add_test(NAME mofft_correctness COMMAND mofft_correctness)

  add_executable(mofft_planning_correctness tests/test_planning.c)
  target_link_libraries(mofft_planning_correctness PRIVATE mofft)
  target_compile_options(mofft_planning_correctness PRIVATE -O2)
  add_test(NAME mofft_planning_correctness COMMAND mofft_planning_correctness)

  add_executable(mofft_plan_recipe_correctness tests/test_plan_recipe.c)
  target_link_libraries(mofft_plan_recipe_correctness PRIVATE mofft)
  target_compile_options(mofft_plan_recipe_correctness PRIVATE -O2)
  add_test(NAME mofft_plan_recipe_correctness COMMAND mofft_plan_recipe_correctness)

  if(MOFFT_BUILD_KERNEL_PORTFOLIO)
    add_executable(mofft_binary_banks tests/test_binary_banks.c)
    target_link_libraries(mofft_binary_banks PRIVATE mofft)
    target_compile_options(mofft_binary_banks PRIVATE -O2)
    add_test(NAME mofft_binary_banks COMMAND mofft_binary_banks)
    set_tests_properties(mofft_binary_banks PROPERTIES TIMEOUT 600)
  endif()

  add_test(NAME mofft_compiler_tests
    COMMAND ${CMAKE_COMMAND} -E env
            "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/compiler"
            ${Python3_EXECUTABLE} -m unittest discover
            -s "${CMAKE_CURRENT_SOURCE_DIR}/tests" -p "test_compiler.py")

  add_test(NAME mofft_portfolio_compiler_tests
    COMMAND ${CMAKE_COMMAND} -E env
            "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/compiler"
            ${Python3_EXECUTABLE} -m unittest discover
            -s "${CMAKE_CURRENT_SOURCE_DIR}/tests" -p "test_portfolio.py")

  add_test(NAME mofft_binary_workflow_tests
    COMMAND ${Python3_EXECUTABLE} -m unittest discover
            -s "${CMAKE_CURRENT_SOURCE_DIR}/tests" -p "test_binary_workflow.py")

  add_custom_target(mofft_validate_candidates
    COMMAND ${CMAKE_COMMAND} -E env
            "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/compiler"
            ${Python3_EXECUTABLE} -m mofft_compiler.validate
            --root "${CMAKE_CURRENT_SOURCE_DIR}"
    DEPENDS
      compiler/mofft_compiler/validate.py
      compiler/mofft_compiler/emitter.py
      compiler/mofft_compiler/patterns.py
      tests/candidate_entry.S
    VERBATIM)
  if(MOFFT_VALIDATE_ALL_CANDIDATES)
    add_test(NAME mofft_all_candidate_kernels
      COMMAND ${CMAKE_COMMAND} -E env
              "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/compiler"
              ${Python3_EXECUTABLE} -m mofft_compiler.validate
              --root "${CMAKE_CURRENT_SOURCE_DIR}")
  endif()
endif()

if(MOFFT_BUILD_TOOLS)
  add_executable(mofft_microbench tools/microbench.c tools/microkernels.S)
  # Keep the C harness outside streaming SVE. AppleClang may otherwise emit
  # `addvl` in ordinary functions, which traps on Apple Silicon before SMSTART.
  set_source_files_properties(tools/microbench.c PROPERTIES
    COMPILE_OPTIONS "-O3;-march=armv9.2-a+nosve")
  set_source_files_properties(tools/microkernels.S PROPERTIES
    COMPILE_OPTIONS
      "-march=armv9.2-a+sme+sme2+sme-f64f64;-mno-implicit-sme")
  target_link_libraries(mofft_microbench PRIVATE m)

  add_executable(mofft_benchmark tools/benchmark.cpp)
  target_link_libraries(mofft_benchmark PRIVATE mofft)
  target_compile_features(mofft_benchmark PRIVATE cxx_std_17)
  target_compile_definitions(mofft_benchmark PRIVATE
    "MOFFT_COMPILER_INFO=\"${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_COMPILER_VERSION}\"")
  if(MOFFT_WITH_FFTW)
    find_path(FFTW_INCLUDE_DIR fftw3.h REQUIRED)
    find_library(FFTW_DOUBLE_LIBRARY fftw3 REQUIRED)
    find_library(FFTW_FLOAT_LIBRARY fftw3f REQUIRED)
    target_include_directories(mofft_benchmark PRIVATE "${FFTW_INCLUDE_DIR}")
    target_link_libraries(mofft_benchmark PRIVATE
      "${FFTW_DOUBLE_LIBRARY}" "${FFTW_FLOAT_LIBRARY}")
    target_compile_definitions(mofft_benchmark PRIVATE MOFFT_HAVE_FFTW=1)
  endif()
  # This check is intentionally separate from the timing path.  It executes
  # the complete public overall-size suite and is enabled by ctest only for a
  # native build.  iOS builds expose the same --correctness mode to the app
  # harness, but cannot execute an iOS binary through ctest on the host.
  if(MOFFT_BUILD_TESTS AND NOT CMAKE_SYSTEM_NAME STREQUAL "iOS")
    add_test(NAME mofft_overall_correctness
      COMMAND mofft_benchmark --correctness --precision both)
    set_tests_properties(mofft_overall_correctness PROPERTIES TIMEOUT 600)
  endif()

  add_executable(mofft_plan_search tools/plan_search.cpp)
  target_link_libraries(mofft_plan_search PRIVATE mofft)
  target_compile_features(mofft_plan_search PRIVATE cxx_std_17)
  target_compile_definitions(mofft_plan_search PRIVATE
    "MOFFT_MANIFEST_PATH=\"${MOFFT_GENERATED_DIR}/manifest.json\"")

  add_custom_target(mofft_check_codegen
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/tools/check_codegen.py"
            --archive "$<TARGET_FILE:mofft>"
            --microbench "$<TARGET_FILE:mofft_microbench>"
            --manifest "${MOFFT_GENERATED_DIR}/manifest.json"
    DEPENDS mofft mofft_microbench tools/check_codegen.py
    VERBATIM)
  if(MOFFT_BUILD_TESTS)
    add_test(NAME mofft_codegen
      COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/tools/check_codegen.py"
              --archive "$<TARGET_FILE:mofft>"
              --microbench "$<TARGET_FILE:mofft_microbench>"
              --manifest "${MOFFT_GENERATED_DIR}/manifest.json")
  endif()
endif()

install(TARGETS mofft EXPORT MoFFTTargets
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")
install(FILES include/mofft.h DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(FILES "${MOFFT_BUILD_INFO}"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/mofft")
install(FILES LICENSE
        DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/MoFFT")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/NOTICE")
  install(FILES NOTICE
          DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/MoFFT")
endif()
install(EXPORT MoFFTTargets NAMESPACE MoFFT::
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/MoFFT")
configure_package_config_file(
  cmake/MoFFTConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/MoFFTConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/MoFFT")
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/MoFFTConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}" COMPATIBILITY SameMajorVersion)
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/MoFFTConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/MoFFTConfigVersion.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/MoFFT")
