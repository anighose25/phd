include_directories(${ASCEND_INCLUDE_DIR})
link_directories(${ASCEND_LIB_DIR})
add_executable (adas_scheduler  AdasScheduler.cpp)
target_link_libraries (adas_scheduler LINK_PUBLIC ascend -lOpenCL)
