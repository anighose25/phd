include_directories(${ASCEND_INCLUDE_DIR})
link_directories(${ASCEND_LIB_DIR})
add_executable (execute_kernel ExecuteKernel.cpp)
target_link_libraries (execute_kernel LINK_PUBLIC ascend -lOpenCL)
