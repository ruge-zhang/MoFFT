cmake_minimum_required(VERSION 3.24)
project(MoFFTConsumerExample LANGUAGES C)

find_package(MoFFT 0.4 CONFIG REQUIRED)

add_executable(mofft-demo main.c)
target_compile_features(mofft-demo PRIVATE c_std_99)
target_link_libraries(mofft-demo PRIVATE MoFFT::mofft)
