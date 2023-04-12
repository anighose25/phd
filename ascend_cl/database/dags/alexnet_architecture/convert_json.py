import sys
import os
from os import listdir
from os.path import join
import json
from copy import deepcopy   
from decimal import *
from collections import OrderedDict

def check_for_func(S):
    '''
    Function to check if a string has a function or not
    '''

    if '(' in S and ')' in S:
        return 1
    return 0

def replace_in_func(S, a, b):
    '''
    Replacing value in a string having a function
    '''
    
    x = S.split('(')
    y = x[1].split(')')
    y[0] = y[0].replace(a, b)
    return x[0] + '(' + y[0] + ')' + y[1]

def replace_iterative(json_file, a, b):
    '''
    Iterative function for replacement of symbolic variables

    dictionary: json dictionary in which sym values have to be replaced
    a: sym variable which has to be replaced
    b: value which sym variable has to be replaced with
    '''

    a = str(a)
    b = str(b)

    for key in json_file:
        if key == "inputBuffers" or key == "outputBuffers" or key == "ioBuffers" or key == "varArguments" or key == "localArguments":
            for item in json_file[key]:
                for k in item:
                    if type(item[k]) in [str, unicode] and k != "type":
                        if check_for_func(item[k]):
                            item[k] = replace_in_func(item[k], a, b)
                        else:
                            item[k] = item[k].replace(a,b)
                            try:
                                item[k] = str(eval(item[k]))
                            except NameError:
                                pass    
        elif key == "macros_values":
            for item in json_file[key]:
                if type(json_file[key][item]) in [str]:
                    if check_for_func(json_file[key]):
                        json_file[key][item] = replace_in_func(json_file[key][item], a, b)
                    else:    
                        json_file[key][item] = json_file[key][item].replace(a, b)
        elif key == "globalWorkSize" or key == "localWorkSize":
            if check_for_func(json_file[key]):
                json_file[key] = replace_in_func(json_file[key], a, b)
            else:
                json_file[key] = json_file[key].replace(a, b)
            try:
                worksize_list = eval(json_file[key]) 
                json_file[key] = str(worksize_list)   
            except NameError:
                pass

def create_dag(info_folder,dag_file,output_folder=None,partition=-1):
    dag_info = open(dag_file,'r').readlines()
    counter = 0

    task_map = {}
    index_map = {}
    task_symvar_map = {}
    while dag_info[counter]!='---\n':
        line = dag_info[counter].strip("\n")
        key,value,symvar = line.split(" ")
        file_name_edit = value.split(".")
        index_map[int(key)] = file_name_edit[0][-1]
        file_name_edit[0] = file_name_edit[0][0:-1]
        value = file_name_edit[0] +"."+file_name_edit[1]
        task_map[int(key)] = value
        task_symvar_map[int(key)]=eval(symvar)
        counter +=1
    counter +=1

    json_files = [join(info_folder,f) for f in listdir(info_folder)]
    json_dictionary = {}
    dag_json = []

    for f in listdir(info_folder):
        if f.endswith('.json') and (f in task_map.values()):
            filename = join(info_folder,f)
            with open(filename,'r') as g:
                json_dictionary[f]=json.loads(g.read(), object_pairs_hook=OrderedDict)

    if not os.path.exists('output'):
        os.makedirs('output')    

    for t in task_map:
        json_file = deepcopy(json_dictionary[task_map[t]])
        symbolicVariables = task_symvar_map[t]

        for sym,val in symbolicVariables.items():
            replace_iterative(json_file,sym,val)

        file_name_edit = task_map[t].split(".")
        file_name_edit[0] = file_name_edit[0] + index_map[t]
        task_map[t] = file_name_edit[0] +"."+file_name_edit[1]

        with open("output/"+task_map[t],'w+') as g:
            json.dump(json_file,g,indent=2)        

if __name__ == "__main__":

    info_folder = sys.argv[1]
    dag_file = sys.argv[2]

    create_dag(info_folder=info_folder, dag_file=dag_file)

