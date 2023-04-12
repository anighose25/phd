include_directories(${ASCEND_INCLUDE_DIR})
link_directories(${ASCEND_LIB_DIR})
add_executable (kernel_callback DispatchKernelCallback.cpp)
target_link_libraries (kernel_callback LINK_PUBLIC ascend -lOpenCL)