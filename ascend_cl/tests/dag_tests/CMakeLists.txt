include_directories(${ASCEND_INCLUDE_DIR})
link_directories(${ASCEND_LIB_DIR})
add_executable (testdag  DAGTest.cpp)
target_link_libraries (testdag LINK_PUBLIC ascend -lOpenCL -lpthread)
