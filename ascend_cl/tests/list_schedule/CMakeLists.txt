include_directories(${ASCEND_INCLUDE_DIR})
link_directories(${ASCEND_LIB_DIR})
add_executable (listschedule  ListScheduling.cpp)
target_link_libraries (listschedule LINK_PUBLIC ascend -lOpenCL)
