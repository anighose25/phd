include_directories(${ASCEND_INCLUDE_DIR})
link_directories(${ASCEND_LIB_DIR})
add_executable (testtaskdag  TaskDAGTest.cpp)
target_link_libraries (testtaskdag LINK_PUBLIC ascend -lOpenCL)
