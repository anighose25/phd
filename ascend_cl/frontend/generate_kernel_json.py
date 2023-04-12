import os
import sys
import subprocess
import json
from convert_json import replace_iterative,replace_in_func,check_for_func
from copy import deepcopy

def create_dag(kernel,dag_file,output_file,partition=-1):
    dag_info = open(dag_file,'r').readlines()
    counter = 0

    task_map = {}
    task_symvar_map = {}
    while dag_info[counter]!='---\n':
        line = dag_info[counter].strip("\n")
        key,value,symvar = line.split(" ")
        task_map[int(key)] = value
        task_symvar_map[int(key)]=eval(symvar)
        counter +=1
    counter +=1

    
    json_dictionary = {}
    
    with open("./database/info/"+kernel) as f:
		json_dictionary[kernel] = json.load(f)
    

    if not os.path.exists('output'):
        os.makedirs('output')    

    for t in task_map:
        json_file = deepcopy(json_dictionary[task_map[t]])
        json_file["partition"]=partition
        symbolicVariables = task_symvar_map[t]
        for sym,val in symbolicVariables.items():
            replace_iterative(json_file,sym,val)

        with open(output_file,'w+') as g:
            json.dump(json_file,g,indent=2)        


if __name__=='__main__':
	lims = [1024,32,8]
	GLOBAL_WORK_SIZES = [8192*2]
	local_work_sizes = [16,32]
	total_runs = 5
	kernel=sys.argv[1]
	partition = int(sys.argv[2])
	global_work_size=sys.argv[3]
	local_work_size=sys.argv[4]

	if kernel.endswith('json'):
		print "Generating ",kernel,partition,global_work_size,local_work_size
		

        
        
		dump_file_name = kernel[:-5]+"_"+str(partition)+"_"+str(global_work_size)+"_"+str(local_work_size)
		dump_folder_name = "./database/info/"+kernel[:-5]
		output_file=dump_folder_name + "/"+ dump_file_name+".json"
		if not os.path.exists(dump_folder_name):
			os.makedirs(dump_folder_name)
		# print dump_file_name
		to_write = "0 {} {}\"dataset\":{},\"n_chunks\":1,\"n_chunks\":1,\"localWorkSize\":{},\"partition\":{}{}\n---\n---\n".\
		format(kernel,"{",global_work_size,local_work_size,partition,"}")
		if kernel == "FFC_sans_bias.json" or kernel == "coalesced_gemm.json":
			to_write = "0 {} {}\"m1\":{},\"p1\":{},\"n1\":{},\"n_chunks\":1,\"wpt\":1,\"localWorkSize\":{},\"TS\":{},\"partition\":{}{}\n---\n---\n".\
			format(kernel,"{",global_work_size,global_work_size,global_work_size,local_work_size,local_work_size,partition,"}")
		# print to_write
		with open("./database/dags/dag.graph","w") as f:
			f.write(to_write)
		# print kernel, output_file	
		create_dag(kernel,"./database/dags/dag.graph",output_file,partition)
			# print(to_write)
			# subprocess.call("python scheduling/multiple_dag_devices.py -f ./dag_info/profile/ -ng 1 -nc 1 -rc -fdp {}/{}.json".format(dump_folder_name,dump_file_name),shell=True)
