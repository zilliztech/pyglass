file(GLOB TEST_SRCS l2_test.cc)

add_executable(test_helpa ${TEST_SRCS})
target_link_libraries(test_helpa helpa gtest gtest_main)
