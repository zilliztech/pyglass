cmake_minimum_required (VERSION 3.1)
project(glass LANGUAGES CXX)

add_library(glass INTERFACE)
target_include_directories(glass INTERFACE .) 

set(CXX_STANDARD 20)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

SET(CMAKE_CXX_FLAGS  "-Wall -Wextra -O3 -lrt -std=c++20 -march=native -fpic -fopenmp -ftree-vectorize -fno-exceptions -fno-rtti" )

add_executable(main examples/main.cc)
target_link_libraries(main glass)
