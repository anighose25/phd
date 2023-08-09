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

    pickle_dumps = {}
    nD =0
    for num_devices in range(4,5):

        dag = DC.create_dag_from_file(sys.argv[2], 0, global_map, ex_map)
        DC.dump_graph_ids(dag, "list_dag_ids.png")
        DC.dump_graph_class(dag, "list_class_dag_ids.png")

        dag_objects = [dag]
        SE = ScheduleEngine(num_devices, num_devices, "blevel", ex_map)
        SE.schedule_workflows(dag_objects, "list", 0, -1)
        print dag.accuracy
	print "MAKESPAN: ", num_devices, SE.time_stamp
        dag_name = "list_dag_timings.png"
        DC.dump_graph_start_finish_times(dag, dag_name)
        # dag_name = heuristic + "_" + str(num_devices)+ "_" + sys.argv[2][:-5]+"png"
        #DC.dump_graph_start_finish_times(dag, dag_name)
