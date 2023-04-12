#include <core.h>
#include <Kernel.h>
#include <ScheduleEngine.h>

#include <HostArray.h>
#include <FreeTreeAllocator.h>	
#include <Buffer.h>


FreeTreeAllocator allocator(static_cast<std::size_t>(1e8));	
bool static_hostarray_allocation = true;

int main()
{
	
	std::vector<std::string> dag_names;
	dag_names.push_back(std::string("edlenet_32"));
	printf("Instatiating ScheduleEngine\n");	

	ScheduleEngine *SE=new ScheduleEngine(Vendor::ARM_GPU,Vendor::ARM_CPU,4,2);
	SE->print_all_device_info();
 	SE->initialize_dags(dag_names);
 	SE->print_all_dag_info();
	SE->setup_dags();
	//SE->schedule_dags();
 	delete SE;
 	
 	printf("Main finished\n");
 	return 0;
}
