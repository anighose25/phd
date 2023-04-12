import subprocess
from os import listdir
from os.path import join
import json

info_folder = "database/info/"

kernel_name = "atax_kernel1.json"


json_files = [join(info_folder,f) for f in listdir(info_folder)  if f.endswith('.json') ]
json_dictionary = {}

for f in json_files:
    with open(f,'r') as g:
        json_dictionary[f]=json.loads(g.read())

global_work_size = [16 ,32, 64, 128, 256, 512, 1024]
local_work_size = [16 ,32, 64, 128, 256]
partition = [0, 10]

print json_files[0].split('/')[2]

for File in json_files:
    flag = 0
    for key in json_dictionary[File]:
        if key == "workDimension":
            if json_dictionary[File][key] == 2:
                flag = 1

    for gws in global_work_size:
        for lws in local_work_size:
            for part in partition:
                if flag == 0:
                    if lws <= gws:
                        subprocess.call(["python", "frontend/generate_kernel_json.py", "{}".format(File.split('/')[2]), "{}".format(part), "{}".format(gws), "{}".format(lws)])
                else:
                    if lws <= gws:
                        if gws*gws <= 1024:
                            subprocess.call(["python", "frontend/generate_kernel_json.py", "{}".format(File.split('/')[2]), "{}".format(part), "{}".format(gws), "{}".format(lws)])
                