include_directories(${ASCEND_INCLUDE_DIR})
link_directories(${ASCEND_LIB_DIR})
add_executable (profile_tc  ProfileTaskComponents.cpp)
target_link_libraries (profile_tc LINK_PUBLIC ascend -lOpenCL)
