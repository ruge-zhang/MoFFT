cmake_minimum_required(VERSION 3.24)
project(MoFFTInstalledConsumer LANGUAGES C)

find_package(MoFFT 0.4 CONFIG REQUIRED)

add_executable(mofft-installed-consumer main.c)
target_link_libraries(mofft-installed-consumer PRIVATE MoFFT::mofft)

set(MOFFT_CONSUMER_REQUIRE_WISDOM_SIZE "0" CACHE STRING
    "Require a PRETUNED plan-wisdom hit at this size (0 disables the check)")
if(NOT MOFFT_CONSUMER_REQUIRE_WISDOM_SIZE MATCHES "^[0-9]+$")
  message(FATAL_ERROR "MOFFT_CONSUMER_REQUIRE_WISDOM_SIZE must be an integer")
endif()
target_compile_definitions(mofft-installed-consumer PRIVATE
  MOFFT_CONSUMER_REQUIRE_WISDOM_SIZE=${MOFFT_CONSUMER_REQUIRE_WISDOM_SIZE})

