include_directories(${ASCEND_INCLUDE_DIR})
link_directories(${ASCEND_LIB_DIR})
add_executable (list_schedule  ListScheduling.cpp)
target_link_libraries (list_schedule LINK_PUBLIC ascend -lOpenCL)
