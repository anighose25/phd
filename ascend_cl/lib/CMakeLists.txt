
# set include dir
include_directories(${ASCEND_INCLUDE_DIR})

#link_directories (${ASCEND_LIB_DIR})

# set files used to build library

set (ASCEND_SOURCES
	  core.cpp
	  Events.cpp
	FreeTreeAllocator.cpp
	RBTree.cpp
	GpuBufferManager.cpp	
      HostArray.cpp
      Buffer.cpp
      KernelArgument.cpp
      Kernel.cpp
      DAG.cpp
      ScheduleEngine.cpp
      )

#add_library (ascend ${ASCEND_SOURCES})
#target_link_libraries (SkelCLCore ${SKELCL_CORE_COMMON_LIBS})


# specify library target

add_library (ascend SHARED ${ASCEND_SOURCES})
target_include_directories (ascend PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ascend -lOpenCL -lpthread)
install(TARGETS ascend DESTINATION lib)



