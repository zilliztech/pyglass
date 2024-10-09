cmake_minimum_required (VERSION 3.1)
project(glass LANGUAGES CXX)

add_library(glass INTERFACE)
target_include_directories(glass INTERFACE .) 

set(CXX_STANDARD 20)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

SET(CMAKE_CXX_FLAGS  "-Wall -Wextra -O3 -lrt -std=c++20 -march=native -fpic -fopenmp -ftree-vectorize -fno-exceptions -fno-rtti" )

add_executable(main main.cpp)
target_link_libraries(main glass)

add_executable(nc_distribute NC_distributed.cpp)
target_link_libraries(nc_distribute glass)

add_executable(sq4_rerank sq4_rerank.cpp)
target_link_libraries(sq4_rerank glass)