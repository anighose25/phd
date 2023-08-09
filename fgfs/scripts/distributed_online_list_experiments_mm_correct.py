from simulate import *
from mpi4py import MPI

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    global_map = {}
    global_file_list = open(sys.argv[1], "r").readlines()
    global_map = make_dict(global_file_list)
    kernel_info_list = [obtain_kernel_info(key) for key in global_map]
    ex_cpu = extract_dict_from_pickle(os.path.join(os.path.dirname(__file__), 'execCPU.pickle'))
    ex_gpu = extract_dict_from_pickle(os.path.join(os.path.dirname(__file__), 'execGPU.pickle'))
    ex_map = (ex_cpu, ex_gpu)
    file_names = []
    
    #OLD RATES
    # online_file = "OnlineMixtureModelConfigurations/online_configuration_mm_rank" +str(rank) + ".stats"
    #NEW RATES
    online_file = "OnlineMixtureModelConfigurationsCorrect/online_configuration_mm_correct_rank" +str(rank) + ".stats"
    # file_name = "NewStats/dump_graphs_new_"+str(rank)+".stats"
    #makespan_list_file = open("Makespans/makespans_list_" +str(rank) + ".stats", 'w')
    
    #OLD RATES
    # makespan_list_file = open("OnlineMakespansMM/"+"onlinemakespans_list_" +str(rank) + ".stats", 'w')
    #NEW RATES
    makespan_list_file = open("OnlineMakespansMMCorrect/"+"onlinemakespans_correct_list_" +str(rank) + ".stats", 'w')
    
    errors_lc_file = open("ERRORS/error_lc_" + str(rank)+ ".stats", 'w')
    configurations = open(online_file, 'r').readlines()
    DC = DAGCreator()
    contraction_depth = sys.argv[2]
    # accuracy_lc_file = open("Makespans/accuracy_lc_" +str(rank) + ".stats", 'w')

    for rank_measure in ["blevel"]:
        
        for config in configurations: 
            try:
                config=config.strip("\n")
                num_devices = int(config.split("_")[4])
                config_values = "OnlineMixtureModelConfigurationsCorrect/" + config

                print config_values,num_devices
                
                SE = ScheduleEngine(num_devices, num_devices, rank_measure, ex_map,True,global_map)
                SE.online_setup_configuration(config_values)
                SE.schedule_workflows([], "list", 0, -1)
                # config_file = config_values.split("/")[1]
                SE.calculate_avg_makespan()
                dump_online_makespan_file = "list_individual_makespan_"+config
                print dump_online_makespan_file
                SE.dump_online_makespans("OnlineMakespansMMCorrect/" + dump_online_makespan_file)
                filename_list = rank_measure + "_" + str(contraction_depth) + "_list" + "_" + config[:-6]
                makespan_list_file.write(filename_list + ": " + str(SE.time_stamp) + "," + str(SE.avg_makespan) + "\n")
                # accuracy_lc_file.write(filename_lc + ": " + str(dag_lc.accuracy) + "\n")
            except:
                config_values = "OnlineMixtureModelConfigurationsCorrect/" + config
                errors_lc_file.write(config_values+  " " + str(num_devices) +  "\n")
                print "FAILED !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!GRAPH NAME", config_values


#makespan_list_file.close()
makespan_list_file.close()
# accuracy_lc_file.close()
errors_lc_file.close()

    
