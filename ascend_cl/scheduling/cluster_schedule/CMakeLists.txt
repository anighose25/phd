include_directories(${ASCEND_INCLUDE_DIR})
link_directories(${ASCEND_LIB_DIR})
add_executable (cluster_schedule  ClusterScheduling.cpp)
target_link_libraries (cluster_schedule LINK_PUBLIC ascend -lOpenCL)
