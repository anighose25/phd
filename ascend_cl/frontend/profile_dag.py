import sys
import sys
import os
from os import listdir
from os.path import join
import subprocess
import json 

dag_folder = "database/dags/"

def extract_file_name(f):
    f = f.split("/")
    name = f[len(f)-1]
    return name.replace(".json", "")

def append_to_json(dag_name, device, json_files, cpu_profiled_files, gpu_profiled_files):
    # function to add to JSON 
    def write_json(data, filename='data.json'): 
        with open(filename,'w') as f: 
            json.dump(data, f, indent=4) 
        
    for f in json_files:    
        with open(f) as json_file: 
            data = json.load(json_file) 
            
            # python object to be appended 
            name = extract_file_name(f)

            for cpu_profiles in cpu_profiled_files:
                if name in cpu_profiles:
                    cpu_profile_file_path = cpu_profiles
                    break

            for gpu_profiles in gpu_profiled_files:
                if name in gpu_profiles:
                    gpu_profile_file_path = gpu_profiles
                    break    
            
            with open(cpu_profile_file_path) as fd:
                y = json.load(fd)
            
            with open(gpu_profile_file_path) as fdd:
                x = json.load(fdd)
            

            for keys in y:
                y["cpu_profile"] = y.pop(keys)

            for keys in x:
                x["gpu_profile"] = x.pop(keys)    
            # appending data to emp_details  
            data.update(y)
            data.update(x)

        with open("profiled_json/" + dag_name + "/" + extract_file_name(f)+ ".json", 'w') as g:
            json.dump(data,g,indent=2)  
  
if __name__ == "__main__":
    dag_name = sys.argv[1]
    profile_type = sys.argv[2]

    json_files_location = join(join(dag_folder, dag_name), "output")
    json_files = [join(json_files_location, f) for f in listdir(json_files_location)]
    
    try:
        os.mkdir(join("profiling", dag_name))
    except:
        pass

    try:
        os.mkdir(join(join("profiling", dag_name), "gpu"))
    except:
        pass

    try:
        os.mkdir(join(join("profiling", dag_name), "cpu"))
    except:
        pass

    try:
        os.mkdir("./profiled_json")
    except:
        pass

    try:
        os.mkdir(join("./profiled_json/", dag_name))
    except:
        pass
    
    devices = ["gpu", "cpu"]
    for f in json_files:    
        for d in devices:
            subprocess.call(["sudo","taskset","-c","4-7","./build/scheduling/execute_kernel/execute_kernel", f, 
            d, join("./profiling/{}/".format(dag_name) + d, extract_file_name(f)+ ".timing"), profile_type])
    
    
    cpu_profiled_files_location = join(join("./profiling", dag_name), "cpu")
    gpu_profiled_files_location = join(join("./profiling", dag_name), "gpu")

    cpu_profiled_files = [join(cpu_profiled_files_location, f) for f in listdir(cpu_profiled_files_location)]
    gpu_profiled_files = [join(gpu_profiled_files_location, f) for f in listdir(gpu_profiled_files_location)]

    append_to_json(dag_name, devices, json_files, cpu_profiled_files, gpu_profiled_files)
