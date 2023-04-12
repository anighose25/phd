include_directories(${ASCEND_INCLUDE_DIR})
link_directories(${ASCEND_LIB_DIR})
add_executable (testrank  RankTest.cpp)
target_link_libraries (testrank LINK_PUBLIC ascend -lOpenCL)
