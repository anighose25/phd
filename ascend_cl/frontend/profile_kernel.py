import os
import sys
import subprocess
import json

if __name__=='__main__':
	lims = [1024,32,8]
	GLOBAL_WORK_SIZES = [8192*2]
	local_work_sizes = [16,32]
	total_runs = 5
	kernel=sys.argv[1]
	if kernel.endswith('json'):
		print "Profiling ",kernel
		with open("./database/info/"+kernel) as f:
			json_source = json.load(f)

        work_dimension = json_source["workDimension"]
        limit_of_local_work_size = lims[work_dimension-1]


	for global_work_size in GLOBAL_WORK_SIZES:
		
		for local_work_size in local_work_sizes:
			if local_work_size > global_work_size:
				break
			if local_work_size > limit_of_local_work_size:
				break
			if work_dimension <=2 and local_work_size <= lims[work_dimension]:
				continue
			for partition in [0,10]:
				for run_number in range(total_runs):
					dump_file_name = kernel[:-5]+"_"+str(partition)+"_"+str(global_work_size)+"_"+str(local_work_size)+"_"+str(run_number)
					dump_folder_name = "./profiling/single/"+kernel[:-5]
					if not os.path.exists(dump_folder_name):
						os.makedirs(dump_folder_name)
					print dump_file_name
					to_write = "0 {} {}\"dataset\":{},\"n_chunks\":1,\"n_chunks\":1,\"localWorkSize\":{},\"partition\":{}{}\n---\n---\n".\
					format(kernel,"{",global_work_size,local_work_size,partition,"}")
					if kernel == "FFC_sans_bias.json" or kernel == "coalesced_gemm.json":
						to_write = "0 {} {}\"m1\":{},\"p1\":{},\"n1\":{},\"n_chunks\":1,\"wpt\":1,\"localWorkSize\":{},\"TS\":{},\"partition\":{}{}\n---\n---\n".\
						format(kernel,"{",global_work_size,global_work_size,global_work_size,local_work_size,local_work_size,partition,"}")
					print to_write
					with open("./dag_info/profile/dag.graph","w") as f:
						f.write(to_write)
					print(to_write)
					subprocess.call("python scheduling/multiple_dag_devices.py -f ./dag_info/profile/ -ng 1 -nc 1 -rc -fdp {}/{}.json".format(dump_folder_name,dump_file_name),shell=True)
	import time
	print "Sleeping for 1 minute"
	time.sleep(60)
