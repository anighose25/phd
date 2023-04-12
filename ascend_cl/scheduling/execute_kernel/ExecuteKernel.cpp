#include <Kernel.h>
#include <ScheduleEngine.h>
#include <core.h>
#include <pthread.h>

#include <FreeTreeAllocator.h>

bool profile_engine=false;
FreeTreeAllocator allocator(static_cast<std::size_t>(1e9));	
bool static_hostarray_allocation = true;
    
int main(int argc, char *argv[]) {
    //cpu_set_t cpuset;
    //pthread_t thread_id_scheduler = pthread_self(); 
    //CPU_ZERO(&cpuset);
    //CPU_SET(2, &cpuset);    //TODO: Set according to your platform
    //pthread_setaffinity_np(thread_id_scheduler, sizeof(cpu_set_t), &cpuset);


    if (argc < 4)
        printf(
            "Program must be provided with four arguments (KernelName, "
            "PreferredDevice, TimingDumpFilename Mode(interference/normal)\n");


    std::vector<std::string> kernel_names;
    kernel_names.push_back(std::string(argv[1]));
    std::string device_type{argv[2]};
    std::string mode{argv[4]};
    // printf("Instatiating ScheduleEngine\n");

    if (mode == "interference")
        kernel_names.push_back(
            std::string{"./database/info/microkernel.json"});

    ScheduleEngine *SE =
        new ScheduleEngine(Vendor::ARM_GPU, Vendor::ARM_CPU, 4, 2);

    SE->initialize_kernels(kernel_names);

    Kernel *k = SE->get_kernel_by_name(kernel_names[0].c_str());
    Kernel *m = NULL;
    if (device_type == "cpu")
        k->set_preferred(Vendor::ARM_CPU);
    else if (device_type == "gpu")
        k->set_preferred(Vendor::ARM_GPU);

    if (mode == "interference") {
        m = SE->get_kernel_by_name(kernel_names[1].c_str());
        if (device_type == "cpu")
            m->set_preferred(Vendor::ARM_GPU);
        else
            m->set_preferred(Vendor::ARM_CPU);
    }

    //  SE->print_all_kernel_info();
    SE->setup_kernels();
    SE->copy_persistent_buffers();
    printf("Persistent copy done\n");
    std::thread scheduler(&ScheduleEngine::schedule_kernels, SE);
    scheduler.join();
    SE->profile_kernels(argv[3]);
    
    //    k->print_static_device_profiles();
    delete SE;

    printf("Main finished\n");
    return 0;
}
