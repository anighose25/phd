from simulate import *
from mpi4py import MPI
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':16})


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
    
    accuracy_file_name = "GraphsNewAccuracy/rank_accuracy_file"+str(rank)+".stats"
    accuracy_file_contents = open(accuracy_file_name, 'r')
    accuracy_values = []
    for line in accuracy_file_contents.readlines():
        accuracy_values.append(float(line.strip("\n").split(':')[1].strip(" ")))
    
    
    global_accuracy_values = comm.gather(accuracy_values,root=0)
    
       
    if rank == 0:
        flattened_list = [item for sublist in global_accuracy_values for item in sublist]
        
        with open('graphsnew_accuracy_ml.stats', 'w') as f:
            print >> f, flattened_list
        
        print min(flattened_list), max(flattened_list)
        print len(flattened_list), np.mean(flattened_list)
        print sum(i>=0.9 for i in flattened_list) / float(len(flattened_list))
        plt.clf()
        plt.hist(x=flattened_list, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.95)
        plt.xlabel('Accuracy')
        plt.ylabel('Frequency')
        plt.savefig('graphsnew_accuracy_ml_bold.pdf', bbox_inches='tight')
       
    accuracy_file_contents.close()
    

