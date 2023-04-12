include_directories(${ASCEND_INCLUDE_DIR})
link_directories(${ASCEND_LIB_DIR})
add_executable (setest ScheduleEngineTest.cpp)
target_link_libraries (setest LINK_PUBLIC ascend -lOpenCL)