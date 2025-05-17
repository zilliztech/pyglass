find_package(benchmark REQUIRED)
find_package(GTest REQUIRED)

set(DEPEND_LIBS
        benchmark::benchmark
        GTest::gtest_main
        gtest_main
        pthread
        )

add_executable(bench_helpa bench.cc)
target_link_libraries(bench_helpa helpa ${DEPEND_LIBS})