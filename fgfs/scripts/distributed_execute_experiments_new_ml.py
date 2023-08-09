from simulate import *
from mpi4py import MPI

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    global_map = {}
    global_file_list = open(sys.argv[1], "r").readlines()
    contraction_depth = sys.argv[2]
    global_map = make_dict(global_file_list)
    kernel_info_list = [obtain_kernel_info(key) for key in global_map]
    ex_cpu = extract_dict_from_pickle(os.path.join(os.path.dirname(__file__), 'execCPU.pickle'))
    ex_gpu = extract_dict_from_pickle(os.path.join(os.path.dirname(__file__), 'execGPU.pickle'))
    ex_map = (ex_cpu, ex_gpu)
    file_names = []
    file_name = "NewStats/dump_graphs_new_"+str(rank)+".stats"
    #makespan_list_file = open("Makespans/makespans_list_" +str(rank) + ".stats", 'w')
    makespan_lc_file = open("Makespans/makespans_lc_ml_depth" +str(contraction_depth) +"_" +str(rank) + ".stats", 'w')
    errors_lc_file = open("ERRORS/error_lc_" + str(rank)+ ".stats", 'w')
    graphs = open(file_name, 'r').readlines()
    DC = DAGCreator()
    
    accuracy_lc_file = open("Makespans/accuracy_lc_" +str(rank) + ".stats", 'w')
    for num_devices in range(2,5):
        for rank in ["blevel"]:
            
            for graph in graphs: 
                try:

                    graph_name = graph.split(":")[0]
                    
                    # print "GRAPH NAME", graph_name
                    
                    # dag_list = DC.create_dag_from_file(graph_name, 0, global_map, ex_map)
                    # dag_objects_list = [dag_list]
                    # SE_list = ScheduleEngine(num_devices, num_devices, rank, ex_map)
                    # SE_list.schedule_workflows(dag_objects_list, "list", 0, contraction_depth)
                    # graph_file = graph_name.split("/")[1]
                    # filename_list = rank + "_" + "-1" + "_list_" + str(num_devices) + "_" + graph_file[:-6]
                    # makespan_list_file.write(filename_list + ": " + str(SE_list.time_stamp) + "\n")
                    
                    print "GRAPH NAME", graph_name, num_devices

                    dag_lc = DC.create_dag_from_file(graph_name, 0, global_map, ex_map)
                    dag_lc.modify_partition_classes("GraphsNewAccuracy/accuracy_"+graph_name[10:][:-6]+".stats")
                    dag_objects_lc = [dag_lc]
                    SE_lc = ScheduleEngine(num_devices, num_devices, rank, ex_map)
                    SE_lc.schedule_linear_clusters(dag_objects_lc, int(contraction_depth))
                    
                    
                    graph_file = graph_name.split("/")[1]
                    filename_lc = rank + "_" + str(contraction_depth) + "_lc_" + str(num_devices) + "_" + graph_file[:-6]
                    makespan_lc_file.write(filename_lc + ": " + str(SE_lc.time_stamp) + "\n")
                    accuracy_lc_file.write(filename_lc + ": " + str(dag_lc.accuracy) + "\n")
                except:
                    graph_name = graph.split(":")[0]
                    errors_lc_file.write(graph_name +  " " + str(num_devices) +  "\n")
                    print "FAILED !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!GRAPH NAME", graph_name
   
    
    #makespan_list_file.close()
    makespan_lc_file.close()
    accuracy_lc_file.close()
    errors_lc_file.close()

    
