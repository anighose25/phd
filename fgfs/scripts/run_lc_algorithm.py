from simulate import *

if __name__ == '__main__':
    global_map = {}
    global_file_list = open(sys.argv[1], "r").readlines()
    global_map = make_dict(global_file_list)
    kernel_info_list = [obtain_kernel_info(key) for key in global_map]

    ex_cpu = extract_dict_from_pickle(os.path.join(os.path.dirname(__file__), 'execCPU.pickle'))
    ex_gpu = extract_dict_from_pickle(os.path.join(os.path.dirname(__file__), 'execGPU.pickle'))
    ex_map = (ex_cpu, ex_gpu)

    DC = DAGCreator()

    rank = "blevel"
    # graph_file = "expt_3.graph"
    # graph_file = "expt_12385.graph"
    graph_file = sys.argv[2]

    for num_devices in range(4,5):
        print graph_file,
        nD = num_devices
        dag = DC.create_dag_from_file(graph_file, 0, global_map, ex_map)
        dag.modify_partition_classes("GraphsNewAccuracy/accuracy_"+graph_file[10:][:-6]+".stats")
        # dag.print_device_preferences()
        DC.dump_graph_ids(dag, "lc_dag_ids.png")
        DC.dump_graph_class(dag, "lc_class_dag_ids.png")

        
        dag_objects = [dag]

        SE = ScheduleEngine(num_devices, num_devices, rank, ex_map)
        SE.schedule_linear_clusters(dag_objects, 4)
        dag_name = "lc_dag_timings.png"
        DC.dump_graph_start_finish_times(dag, dag_name)
        print "MAKESPAN: ", num_devices, SE.time_stamp
        # dag_name = "lc_"+graph_file[:-6] + ".png"
            # DC.dump_graph_start_finish_times(dag, dag_name)
