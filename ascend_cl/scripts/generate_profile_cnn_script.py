import os
import sys
info_folder = "json/"+sys.argv[1]
json_files=[f for f in os.listdir(info_folder) if f.endswith('.json')]

for kernel in json_files:
	for device in ["cpu","gpu"]:
		timing_file = "./profiling/"+kernel[:-5]+"_" + device + ".timing"
		command = "./build/scheduling/execute_kernel/execute_kernel " + "./database/info/"+kernel+ " " + device + " " + timing_file + " normal"
		print command