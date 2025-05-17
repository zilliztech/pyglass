cmake_minimum_required(VERSION 3.2)
project(helpa LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)            
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_FLAGS "-march=native -O3")

add_library(helpa INTERFACE )
target_include_directories(helpa INTERFACE .)

add_subdirectory(bench)
