import subprocess
from os import listdir
from os.path import join
import json

info_folder = "database/info/"


json_files = [f for f in listdir(info_folder)  if f.endswith('.json') ]
json_dictionary = {}

for f in json_files:
    with open(join(info_folder,f),'r') as g:
        json_dictionary[f]=json.loads(g.read())

global_work_size = [16 ,32, 64, 128, 256, 512, 1024]
local_work_size = [16 ,32, 64, 128, 256]
partition = [0, 10]

#print json_files[0].split('/')[2]
dev_type = ""
for File in json_files:
    flag = 0
    for key in json_dictionary[File]:
        if key == "workDimension":
            if json_dictionary[File][key] == 2:
                flag = 1

    for gws in global_work_size:
        for lws in local_work_size:
            for part in partition:
                if part == 0:
                    dev_type ="cpu"
                else:
                    dev_type = "gpu"
                if flag == 0:
                    if lws <= gws:
                        print "./build/scheduling/execute_kernel/execute_kernel ./database/info/"+File[:-5]+"/"+File[:-5]+"_"+str(part)+"_"+str(gws)+"_"+str(lws)+".json " + dev_type + " timing/"+File[:-5]+"_"+str(part)+"_"+str(gws)+"_"+str(lws)+".timing"
                else:
                    if lws <= gws:
                        if gws*gws <= 1024:
                            print "./build/scheduling/execute_kernel/execute_kernel ./database/info/"+File[:-5]+"/"+File[:-5]+"_"+str(part)+"_"+str(gws)+"_"+str(lws)+".json " +dev_type + " timing/"+File[:-5]+"_"+str(part)+"_"+str(gws)+"_"+str(lws)+".timing"
                        
                