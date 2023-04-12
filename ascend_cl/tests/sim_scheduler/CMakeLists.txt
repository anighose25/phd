include_directories(${ASCEND_INCLUDE_DIR})
link_directories(${ASCEND_LIB_DIR})
add_executable (sim_scheduler  SimScheduler.cpp)
target_link_libraries (sim_scheduler LINK_PUBLIC ascend -lOpenCL)
