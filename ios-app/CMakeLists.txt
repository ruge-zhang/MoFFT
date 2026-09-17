cmake_minimum_required(VERSION 3.24)
set(CMAKE_SYSTEM_NAME iOS CACHE STRING "" FORCE)
set(CMAKE_OSX_DEPLOYMENT_TARGET 16.0 CACHE STRING "" FORCE)
project(mofftbench_ios LANGUAGES C CXX ASM)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(MOFFT_LIBRARY_BUILD ${ROOT}/build-m4 CACHE PATH
    "MoFFT build directory linked into the iOS benchmark app")
set(MOFFT_FFTW_IOS_PREFIX ${ROOT}/build/deps/fftw-3.3.11/install-ios CACHE PATH
    "FFTW install (arm64-ios) linked into the iOS benchmark for the FFTW baseline")
set(MOFFT_IOS_BUNDLE_IDENTIFIER org.mofft.mofftbench CACHE STRING
    "Bundle identifier for the iOS benchmark app")
set(MOFFT_IOS_SEARCH_BUNDLE_IDENTIFIER org.mofft.mofftsearch CACHE STRING
    "Bundle identifier for the iOS plan-search app")
set(MOFFT_IOS_CALIBRATE_BUNDLE_IDENTIFIER org.mofft.mofftcalibrate CACHE STRING
    "Bundle identifier for the iOS calibration app")
add_executable(mofftbench MACOSX_BUNDLE ${ROOT}/tools/benchmark.cpp)
target_include_directories(mofftbench PRIVATE
  ${ROOT}/include ${MOFFT_LIBRARY_BUILD}/generated
  ${MOFFT_FFTW_IOS_PREFIX}/include)
target_compile_definitions(mofftbench PRIVATE MOFFT_HAVE_FFTW=1)
target_compile_definitions(mofftbench PRIVATE
  "MOFFT_COMPILER_INFO=\"${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_COMPILER_VERSION}\"")
target_link_libraries(mofftbench PRIVATE
  ${MOFFT_LIBRARY_BUILD}/libmofft.a
  ${MOFFT_FFTW_IOS_PREFIX}/lib/libfftw3.a
  ${MOFFT_FFTW_IOS_PREFIX}/lib/libfftw3f.a)
set_target_properties(mofftbench PROPERTIES
  OUTPUT_NAME mofftbench
  MACOSX_BUNDLE_INFO_PLIST ${CMAKE_CURRENT_SOURCE_DIR}/Info.plist
  XCODE_ATTRIBUTE_CODE_SIGN_STYLE Automatic
  XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY "Apple Development"
  XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER ${MOFFT_IOS_BUNDLE_IDENTIFIER}
  XCODE_ATTRIBUTE_TARGETED_DEVICE_FAMILY "1,2")

# The plan-search app evaluates the generated plans on the device and emits
# its wisdom JSON on stdout.  The manifest is copied into the application
# bundle so that the app has the same generated-kernel metadata as the linked
# MoFFT library.
set(MOFFT_IOS_MANIFEST ${MOFFT_LIBRARY_BUILD}/generated/manifest.json)
set_source_files_properties(${MOFFT_IOS_MANIFEST} PROPERTIES
  MACOSX_PACKAGE_LOCATION Resources)
add_executable(mofftsearch MACOSX_BUNDLE
  ${ROOT}/tools/plan_search.cpp ${MOFFT_IOS_MANIFEST})
target_include_directories(mofftsearch PRIVATE
  ${ROOT}/include ${MOFFT_LIBRARY_BUILD}/generated)
target_link_libraries(mofftsearch PRIVATE ${MOFFT_LIBRARY_BUILD}/libmofft.a)
set_target_properties(mofftsearch PROPERTIES
  OUTPUT_NAME mofftsearch
  MACOSX_BUNDLE_INFO_PLIST ${CMAKE_CURRENT_SOURCE_DIR}/Info-search.plist
  XCODE_ATTRIBUTE_CODE_SIGN_STYLE Automatic
  XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY "Apple Development"
  XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER ${MOFFT_IOS_SEARCH_BUNDLE_IDENTIFIER}
  XCODE_ATTRIBUTE_TARGETED_DEVICE_FAMILY "1,2")

# The calibration app emits the device microbenchmark JSON on stdout.  The
# host-side calibration tool can therefore consume the output from
# `devicectl --console` without requiring file access inside the app sandbox.
add_executable(mofftcalibrate MACOSX_BUNDLE
  ${ROOT}/tools/microbench.c ${ROOT}/tools/microkernels.S)
set_source_files_properties(${ROOT}/tools/microbench.c PROPERTIES
  COMPILE_OPTIONS "-O3;-march=armv9.2-a+nosve")
set_source_files_properties(${ROOT}/tools/microkernels.S PROPERTIES
  COMPILE_OPTIONS
    "-march=armv9.2-a+sme+sme2+sme-f64f64;-mno-implicit-sme")
target_link_libraries(mofftcalibrate PRIVATE m)
set_target_properties(mofftcalibrate PROPERTIES
  OUTPUT_NAME mofftcalibrate
  MACOSX_BUNDLE_INFO_PLIST ${CMAKE_CURRENT_SOURCE_DIR}/Info-calibrate.plist
  XCODE_ATTRIBUTE_CODE_SIGN_STYLE Automatic
  XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY "Apple Development"
  XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER ${MOFFT_IOS_CALIBRATE_BUNDLE_IDENTIFIER}
  XCODE_ATTRIBUTE_TARGETED_DEVICE_FAMILY "1,2")
