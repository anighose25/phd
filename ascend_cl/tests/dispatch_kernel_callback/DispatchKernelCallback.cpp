#include <core.h>
#include <Kernel.h>
#include <ScheduleEngine.h>
#include <HostArray.h>
#include <FreeTreeAllocator.h>	


FreeTreeAllocator allocator(static_cast<std::size_t>(1e8));	
bool static_hostarray_allocation = true;

int main()
{
	
	std::vector<std::string> kernel_names;
	kernel_names.push_back(std::string("atax1.json"));
	printf("Instatiating ScheduleEngine\n");	

	ScheduleEngine *SE=new ScheduleEngine(Vendor::ARM_GPU,Vendor::ARM_CPU,4,2);
	
 	SE->initialize_kernels(kernel_names);
 	SE->print_all_kernel_info();
	SE->setup_kernels();
	SE->schedule_kernels();
 	
 	delete SE;
 	
 	printf("Main finished\n");
 	return 0;
}
