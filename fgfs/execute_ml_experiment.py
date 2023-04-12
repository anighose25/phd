from simulate import *
from mpi4py import MPI

if __name__ == '__main__':
    
    global_map = {}
    global_file_list = open(sys.argv[1], "r").readlines()
    global_map = make_dict(global_file_list)
    kernel_info_list = [obtain_kernel_info(key) for key in global_map]
    ex_cpu = extract_dict_from_pickle(os.path.join(os.path.dirname(__file__), 'execCPU_new.pickle'))
    ex_gpu = extract_dict_from_pickle(os.path.join(os.path.dirname(__file__), 'execGPU_new.pickle'))
    ex_map = (ex_cpu, ex_gpu)
    graph_name = sys.argv[2]
    DC = DAGCreator()
    dag_lc = DC.create_dag_from_file(graph_name, 0, global_map, ex_map,ml=True)
    dag_lc.print_device_preferences()
    print dag_lc.accuracy
           
    
